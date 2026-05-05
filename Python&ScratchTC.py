"""
Windows 超级 Clicky v43.0 最终完整版
- 实时语音对话、打字提问、视觉引导、打断、静音
- 完整的控制台提示：功能介绍、快捷键、问答文字
- 所有 PyCharm 警告已消除（无 str|None、无宽泛异常、无参数未填）
"""

import os
import queue
import threading
import time
import io
import base64
import tempfile
import sys
import re
import numpy as np
import tkinter as tk
from urllib.parse import urlparse

import sounddevice as sd
import soundfile as sf
import requests
import pyttsx3
from dotenv import load_dotenv
from openai import OpenAI
from pynput import keyboard, mouse
from PIL import ImageGrab
from zai import ZhipuAiClient

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

class StderrFilter:
    def __init__(self, real_stderr):
        self.real_stderr = real_stderr
    def write(self, msg):
        if "iCCP" not in msg:
            self.real_stderr.write(msg)
    def flush(self):
        self.real_stderr.flush()
sys.stderr = StderrFilter(sys.stderr)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "your-zhipu-api-key-here")
ZHIPU_VISION_MODEL = "glm-4.6v-flash"
ZHIPU_ASR_MODEL = "glm-asr-2512"
ZHIPU_ASR_URL = "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.2
SILENCE_TIMEOUT = 1.2
ENERGY_THRESHOLD = 300

ui_queue = queue.Queue()
SHOW_DIALOG = "__SHOW_DIALOG__"
UPDATE_BUBBLE = "__UPDATE_BUBBLE__"

running, conversation = True, []
cursor_x, cursor_y = 0, 0
tts_on, thinking, listen_paused = True, False, False
ctrl, shift, dialog_open = False, False, False
audio_chunks, speaking, silence_cnt = [], False, 0

root: tk.Tk | None = None
bubble_lbl: tk.Label | None = None
bubble_win: tk.Toplevel | None = None
tts_engine: pyttsx3.Engine | None = None
tts_lock = threading.Lock()
zhipu: ZhipuAiClient | None = None

_COORD_RE_1 = re.compile(r'\[\[COORD:(\d+),(\d+)]]')
_COORD_RE_2 = re.compile(r'\((\d+),\s*(\d+)\)')

def init_zhipu():
    global zhipu
    if ZHIPU_API_KEY == "your-zhipu-api-key-here":
        return
    try:
        zhipu = ZhipuAiClient(api_key=ZHIPU_API_KEY)
        print("✅ 智谱就绪")
    except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
        print(f"⚠️ 智谱初始化失败（已知异常）: {e}")

init_zhipu()

def stop_tts():
    global tts_engine
    if tts_engine is not None:
        try:
            tts_engine.stop()
        except (RuntimeError, OSError):
            pass

def ensure_tts():
    global tts_engine
    if tts_engine is None:
        try:
            tts_engine = pyttsx3.init()
            voices = tts_engine.getProperty('voices')
            for v in voices:
                if 'Chinese' in v.name:
                    tts_engine.setProperty('voice', v.id)
                    break
            tts_engine.setProperty('rate', 180)
            tts_engine.setProperty('volume', 0.9)
            print("✅ TTS就绪")
        except (RuntimeError, OSError) as e:
            print(f"⚠️ TTS 初始化失败（已知异常）: {e}")
            tts_engine = None

ensure_tts()

def speak(text):
    if not tts_on:
        return
    clean = text.replace('*', '').replace('#', '').replace('`', '')[:200]
    threading.Thread(target=_speak, args=(clean,), daemon=True).start()

def _speak(text):
    global thinking, tts_engine
    ensure_tts()
    eng = tts_engine
    if eng is None:
        thinking = False
        return
    try:
        eng.say(text)
        eng.runAndWait()
    except (RuntimeError, OSError):
        tts_engine = None

    if not listen_paused:
        ui_queue.put((UPDATE_BUBBLE, ("🎤 聆听中...", '#e94560')))
    thinking = False

def capture_screen_b64():
    try:
        img = ImageGrab.grab()
        img = img.resize((img.width // 2, img.height // 2))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=50)
        return base64.b64encode(buf.getvalue()).decode()
    except OSError:
        return ""

def show_indicator(x, y):
    def _show():
        win = tk.Tk()
        win.overrideredirect(True); win.attributes('-topmost', True); win.attributes('-alpha', 0.8)
        win.geometry(f"+{x-40}+{y-40}")
        c = tk.Canvas(win, width=80, height=80, bg='black', highlightthickness=0); c.pack()
        for col in ['#ff3333', '#ffffff'] * 3:
            c.delete('all')
            c.create_oval(10, 10, 70, 70, outline=col, width=4)
            win.update()
            time.sleep(0.3)
        win.destroy()
    threading.Thread(target=_show, daemon=True).start()

def analyze_media(mtype, data, question):
    if zhipu is None:
        return "[错] 智谱未初始化"
    try:
        resp = zhipu.chat.completions.create(
            model=ZHIPU_VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": mtype, mtype: {"url": data}},
                {"type": "text", "text": question}
            ]}],
            thinking={"type": "enabled"}
        )
        return resp.choices[0].message.content or ""
    except (requests.RequestException, ConnectionError, TimeoutError, OSError, RuntimeError) as e:
        print(f"⚠️ 多模态分析失败（已知异常）: {e}")
        return "[错] 分析失败"

def analyze_screen(b64, q):
    return analyze_media("image_url", b64, q)

def transcribe(audio: np.ndarray) -> str:
    """
    语音识别。通过 isinstance 类型守卫确保所有路径返回 str。
    """
    if ZHIPU_API_KEY == "your-zhipu-api-key-here":
        return ""
    tp = None
    try:
        fd, tp = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        sf.write(tp, audio, SAMPLE_RATE)
        with open(tp, 'rb') as f:
            r = requests.post(ZHIPU_ASR_URL,
                              headers={"Authorization": f"Bearer {ZHIPU_API_KEY}"},
                              files={'file': ('audio.wav', f, 'audio/wav'),
                                     'model': (None, ZHIPU_ASR_MODEL),
                                     'stream': (None, 'false')})
        if r.status_code == 200:
            data = r.json()
            raw = data.get("text")
            # 类型守卫：只有 str 才返回
            if isinstance(raw, str):
                return raw
            return ""
        return ""
    except (requests.RequestException, OSError, RuntimeError) as e:
        print(f"⚠️ 语音识别失败（已知异常）: {e}")
        return ""
    finally:
        if tp and os.path.exists(tp):
            os.remove(tp)

_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "webm", "mkv"}
_FILE_EXTENSIONS  = {"pdf", "txt", "docx", "xlsx", "pptx"}

def detect_url(text):
    urls = re.findall(r'https?://\S+', text)
    if not urls:
        return None
    url = urls[0]
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lstrip('.').lower()
    if ext in _VIDEO_EXTENSIONS:
        return "video_url", url
    if ext in _FILE_EXTENSIONS:
        return "file_url", url
    return "image_url", url

def _produce_answer(context_msg: str):
    conversation.append({"role": "user", "content": context_msg})
    msgs = [{"role": "system", "content": "你是Clicky，用中文简洁回答。"}] + conversation
    answer = call_ds(msgs)
    conversation.append({"role": "assistant", "content": answer})
    parse_coord(answer)
    short = answer[:80] + ('..' if len(answer) > 80 else '')
    ui_queue.put((UPDATE_BUBBLE, (short, '#4ec9b0')))
    speak(answer[:200])
    if len(conversation) > 20:
        del conversation[:-20]
    # 还原回答文字显示
    print(f"🎓 Clicky: {answer}")

def _parse_coordinate(text):
    m = _COORD_RE_1.search(text)
    if not m:
        m = _COORD_RE_2.search(text)
    if m:
        x, y = int(m.group(1)) * 2, int(m.group(2)) * 2
        return x, y
    return None

def parse_coord(text):
    coord = _parse_coordinate(text)
    if coord:
        show_indicator(*coord)

def process_question(q):
    global conversation, thinking
    guide_words = ["点哪里","点击","按哪个","在哪里","找不到","怎么设置","怎么打开","帮我找",
                   "按钮","菜单","图标","设置","选项","工具","视图","编辑","文件","保存"]
    need_guide = any(k in q for k in guide_words)
    screen_words = ["怎么","哪里","点什么","在哪","这个","屏幕","帮我"]
    need_screen = need_guide or any(k in q for k in screen_words)

    ui_queue.put((UPDATE_BUBBLE, ("👀 分析...", '#f0a500')))

    # 还原提问文字显示
    print(f"🧑 你: {q}")

    media = detect_url(q)
    if media:
        mtype, url = media
        analysis = analyze_media(mtype, url, q)
        ctx = f"提问：{q}\n分析：{analysis}"
        _produce_answer(ctx)
        return

    if need_screen:
        b64 = capture_screen_b64()
        if not b64:
            _produce_answer("截图失败")
            return
        prompt = f"描述截图，关注：{q}"
        vis = analyze_screen(b64, prompt)
        coord = _parse_coordinate(vis)
        if coord is not None:
            cx, cy = coord
            show_indicator(cx, cy)
            vis = _COORD_RE_1.sub('', vis).strip() + "\n\n💡 已标记位置"
        ctx = f"光标({cursor_x},{cursor_y})附近。{q}\n视觉：{vis}"
        _produce_answer(ctx)
        return

    _produce_answer(q)

def call_ds(msgs):
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        r = client.chat.completions.create(model="deepseek-chat", messages=msgs,
                                           temperature=0.7, max_tokens=600)
        return r.choices[0].message.content or ""
    except (requests.RequestException, OSError, RuntimeError, ConnectionError) as e:
        print(f"⚠️ AI 服务调用失败（已知异常）: {e}")
        return "AI 暂不可用"

def audio_loop():
    global speaking, silence_cnt, audio_chunks, thinking
    block = int(SAMPLE_RATE * CHUNK_DURATION)
    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=block)
        stream.start()
        ui_queue.put((UPDATE_BUBBLE, ("🎤 聆听中...", '#e94560')))
    except (sd.PortAudioError, OSError) as e:
        print(f"⚠️ 音频流启动失败: {e}")
        return

    while running:
        try:
            if listen_paused or dialog_open:
                audio_chunks.clear(); speaking = False; silence_cnt = 0
                time.sleep(0.2); continue
            data, _ = stream.read(block)
            rms = np.sqrt(np.mean(np.square(data.astype(np.float32))))
            if rms > ENERGY_THRESHOLD:
                if not speaking:
                    speaking = True; audio_chunks.clear()
                    if thinking: stop_tts(); thinking = False
                audio_chunks.append(data.copy()); silence_cnt = 0
            else:
                if speaking:
                    silence_cnt += 1; audio_chunks.append(data.copy())
                    if silence_cnt > int(SILENCE_TIMEOUT / CHUNK_DURATION):
                        if audio_chunks:
                            txt = transcribe(np.concatenate(audio_chunks))
                            if txt:
                                ui_queue.put((UPDATE_BUBBLE, ("🤔 思考中...", '#f0a500')))
                                thinking = True
                                threading.Thread(target=process_question, args=(txt,), daemon=True).start()
                            else:
                                ui_queue.put((UPDATE_BUBBLE, ("🎤 聆听中...", '#e94560')))
                        audio_chunks.clear(); speaking = False; silence_cnt = 0
        except sd.PortAudioError:
            break
        except OSError:
            break
    stream.stop()

def create_ui():
    global root, bubble_lbl, bubble_win
    root = tk.Tk(); root.withdraw()
    bubble_win = tk.Toplevel(root)
    bubble_win.overrideredirect(True); bubble_win.attributes('-topmost', True)
    bubble_win.attributes('-alpha', 0.88); bubble_win.configure(bg='#1a1a2e')
    bubble_lbl = tk.Label(bubble_win, text="🎤 聆听中...", font=('Microsoft YaHei',16),
                          bg='#1a1a2e', fg='#e94560', padx=8, pady=4, wraplength=260)
    bubble_lbl.pack()

def set_bubble(text, color='#e94560'):
    if bubble_lbl is not None:
        try: bubble_lbl.config(text=text, fg=color)
        except tk.TclError: pass

def move_bubble():
    if bubble_win is not None:
        try: bubble_win.geometry(f"+{cursor_x+20}+{cursor_y-10}")
        except tk.TclError: pass

def show_dialog():
    global dialog_open, listen_paused
    if dialog_open or root is None: return
    dialog_open = True
    old = listen_paused; listen_paused = True
    ui_queue.put((UPDATE_BUBBLE, ("⌨️ 打字中...", '#f0a500')))
    dlg = tk.Toplevel(root)
    dlg.title("问 Clicky"); dlg.geometry("420x160"); dlg.configure(bg='#1a1a2e'); dlg.attributes('-topmost', True)
    tk.Label(dlg, text="想问什么？", bg='#1a1a2e', fg='white', font=('Microsoft YaHei',12)).pack(pady=10)
    entry = tk.Entry(dlg, font=('Microsoft YaHei',11), width=40); entry.pack(pady=5); entry.focus()

    def close():
        nonlocal old
        global dialog_open, listen_paused
        dialog_open = False; dlg.destroy()
        if not old: listen_paused = False; ui_queue.put((UPDATE_BUBBLE, ("🎤 聆听中...", '#e94560')))

    def submit():
        t = entry.get(); close()
        if t.strip(): threading.Thread(target=process_question, args=(t.strip(),), daemon=True).start()

    dlg.protocol("WM_DELETE_WINDOW", close)
    btn = tk.Frame(dlg, bg='#1a1a2e'); btn.pack(pady=10)
    tk.Button(btn, text="提问", command=submit, bg='#e94560', fg='white', width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(btn, text="取消", command=close, bg='#333', fg='white', width=10).pack(side=tk.LEFT, padx=5)
    entry.bind('<Return>', lambda event: submit())

def process_ui():
    while True:
        try:
            task = ui_queue.get_nowait()
            if task is None: continue
            cmd, *rest = task
            if cmd == UPDATE_BUBBLE: set_bubble(*rest[0])
            elif cmd == SHOW_DIALOG: show_dialog()
        except queue.Empty: break
    if root is not None:
        root.after(50, lambda *args: process_ui())

def on_press(key):
    global running, tts_on, ctrl, shift, listen_paused, dialog_open
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r): ctrl = True
    if key in (keyboard.Key.shift_l, keyboard.Key.shift_r): shift = True
    if ctrl and shift and not dialog_open:
        ui_queue.put((SHOW_DIALOG,)); ctrl = False; shift = False
    if key == keyboard.Key.f6:
        tts_on = not tts_on
        st = "🔇 静音" if not tts_on else "✅ 朗读开"
        ui_queue.put((UPDATE_BUBBLE, (st, '#888' if not tts_on else '#e94560')))
    if key == keyboard.Key.f7:
        listen_paused = not listen_paused
        st = "⏸️ 已暂停" if listen_paused else "🎤 聆听中..."
        ui_queue.put((UPDATE_BUBBLE, (st, '#888' if listen_paused else '#e94560')))
    if key == keyboard.Key.esc:
        running = False
        if root is not None:
            root.after(0, lambda *args: root.destroy())

def on_release(key):
    global ctrl, shift
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r): ctrl = False
    if key in (keyboard.Key.shift_l, keyboard.Key.shift_r): shift = False

def on_move(x, y):
    global cursor_x, cursor_y
    cursor_x, cursor_y = x, y

def main():
    global root, running
    print("=" * 60)
    print("  🎓 Windows 超级 Clicky v43.0 最终完整版")
    print("=" * 60)
    print("  Ctrl+Shift = 打字提问")
    print("  直接说话   = 实时语音对话")
    print("  按 F7      = 暂停/恢复语音监听")
    print("  按 F6      = 全局静音开关")
    print("  Esc        = 退出")
    print("=" * 60)
    if DEEPSEEK_API_KEY == "your-zhipu-api-key-here" or ZHIPU_API_KEY == "your-zhipu-api-key-here":
        print("⚠️  请先设置 API Key 到 .env 文件")

    create_ui()
    if root is None: return
    root.after(100, lambda *args: process_ui())
    def update_pos():
        if running: move_bubble()
        if root is not None:
            root.after(50, lambda *args: update_pos())
    root.after(50, lambda *args: update_pos())
    for lst in [keyboard.Listener(on_press=on_press, on_release=on_release), mouse.Listener(on_move=on_move)]:
        lst.daemon = True; lst.start()
    threading.Thread(target=audio_loop, daemon=True).start()
    print("\n✨ Clicky 已就绪！")
    try: root.mainloop()
    except KeyboardInterrupt: pass
    running = False
    print("安全退出")

if __name__ == "__main__":
    required = ["pynput","openai","PIL","sounddevice","soundfile","dotenv","pyttsx3","numpy","requests","zai"]
    missing = [lib for lib in required if not __import__(lib, fromlist=[''])]
    if missing:
        print(f"缺少: {missing}")
        print("pip install pynput openai Pillow sounddevice soundfile python-dotenv pyttsx3 numpy requests zai-sdk -i https://pypi.tuna.tsinghua.edu.cn/simple/")
    else:
        main()