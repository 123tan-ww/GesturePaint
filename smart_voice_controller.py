import speech_recognition as sr
import threading
import queue
import logging
from typing import Callable, Optional, Dict, Any
import time

class SmartVoiceController:
    
    def __init__(self, on_result_callback: Optional[Callable] = None):
        """
        初始化语音控制器
        :param on_result_callback: 回调函数，接收(text, start_time, end_time, is_final)参数
        """
        # 语音识别
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # 回调函数
        self.on_result_callback = on_result_callback
        
        # 运行状态
        self.is_listening = False
        self.listen_thread = None
        
        # 音频队列（用于未来扩展）
        self.audio_queue = queue.Queue()
        
        # 识别状态
        self.last_partial_time = 0
        self.partial_text = ""
        
        # 调整环境噪声
        self._adjust_for_ambient_noise()
        
        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("🔊 智能语音控制器初始化完成")
    
    def _adjust_for_ambient_noise(self):
        """调整环境噪声阈值"""
        try:
            with self.microphone as source:
                print("🔊 正在校准环境噪声...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("🔊 校准完成")
        except Exception as e:
            print(f"⚠️ 噪声校准失败: {e}")
    
    def start(self):
        """启动语音识别"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
        self.logger.info("语音识别已启动")
        print("🎤 语音识别已启动，请说话...")
    
    def stop(self):
        """停止语音识别"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1)
        self.logger.info("语音识别已停止")
        print("🔇 语音识别已停止")
    
    def _listen_loop(self):
        """语音监听循环"""
        while self.is_listening:
            try:
                # 从麦克风获取音频
                with self.microphone as source:
                    print("🎤 正在聆听...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                # 识别音频
                try:
                    # 先尝试获取最终结果
                    text = self.recognizer.recognize_google(audio, language='zh-CN', show_all=False)
                    print(f"📝 最终识别: {text}")
                    
                    # 调用回调函数，标记为最终结果
                    if self.on_result_callback:
                        start_time = time.time()
                        end_time = time.time()
                        self.on_result_callback(text, start_time, end_time, is_final=True)
                    
                    # 清除部分结果
                    self.partial_text = ""
                    
                except sr.UnknownValueError:
                    # 听不懂时尝试部分识别（模拟）
                    try:
                        # 部分识别（有些API支持，Google可能不支持）
                        partial = self.recognizer.recognize_google(audio, language='zh-CN', show_all=True)
                        if partial and isinstance(partial, dict) and 'alternative' in partial:
                            alternatives = partial['alternative']
                            if alternatives and len(alternatives) > 0:
                                text = alternatives[0]['transcript']
                                print(f"📝 部分识别: {text}")
                                
                                if self.on_result_callback:
                                    start_time = time.time()
                                    end_time = time.time()
                                    self.on_result_callback(text, start_time, end_time, is_final=False)
                    except:
                        pass
                        
                except sr.RequestError as e:
                    print(f"⚠️ 语音识别服务错误: {e}")
                    
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"⚠️ 监听错误: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'is_listening': self.is_listening,
            'partial_text': self.partial_text
        }