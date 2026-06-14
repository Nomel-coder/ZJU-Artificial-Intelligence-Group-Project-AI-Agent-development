import os
import gradio as gr
from agent import ReActAgent

# 初始化ReAct智能体实例
agent = ReActAgent()

def predict(message, history, workspace_dir):
    """
    对话预测主逻辑，流式返回对话内容与执行轨迹
    :param message: 用户输入文本
    :param history: 历史对话记录
    :param workspace_dir: 工作目录路径
    :yield: 实时更新的对话记录、ReAct执行轨迹日志
    """
    # 校验空输入
    if not message.strip():
        yield history, "请输入有效的问题。"
        return
        
    # 动态设置全局工作目录环境变量
    if workspace_dir and workspace_dir.strip():
        os.environ["AGENT_WORKSPACE_DIR"] = os.path.abspath(workspace_dir.strip())
        
    # 重置智能体状态
    agent.reset()
    trace_log = ""
    
    # 初始化对话历史（适配Gradio 6.0+）
    chat_history = list(history) if history else []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": "思考中..."})
    
    # 首次推送状态提示
    yield chat_history, "正在启动 ReAct 推理闭环..."
    
    # 执行智能体推理，获取分步执行结果
    steps = agent.run(message)
    bot_response = ""
    
    # 遍历每一步推理/工具执行流程，流式更新界面
    for step in steps:
        step_type = step["type"]
        # 拼接不同类型步骤的日志内容
        if step_type == "thought":
            trace_log += f"### 🧠 思考过程\n{step['content']}\n\n"
        elif step_type == "tool_call":
            trace_log += f"### 🔧 调用工具: `{step['tool']}`\n**入参:** `{step['input']}`\n\n"
        elif step_type == "observation":
            trace_log += f"### 📄 执行结果\n```text\n{step['content']}\n```\n\n"
        elif step_type == "final_answer":
            bot_response = step["content"]
            trace_log += f"### ✅ 最终答案\n{bot_response}\n"
        elif step_type == "error":
            trace_log += f"### ❌ 异常报错\n{step['content']}\n\n"
            
        # 实时更新助手展示文本
        current_bot_text = bot_response if bot_response else "正在执行步骤，请看右侧执行轨迹..."
        chat_history[-1] = {"role": "assistant", "content": current_bot_text}
        yield chat_history, trace_log
        
    # 处理无最终答案的异常场景
    if not bot_response:
        chat_history[-1] = {"role": "assistant", "content": "未能得出最终结论，请查看运行轨迹。"}
        yield chat_history, trace_log

# 自定义全局主题：深色轻奢风格，优化配色与字体
theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "Microsoft YaHei", "sans-serif"]
)

# 全局CSS样式
css = """
/* 全局页面基础样式 */
body, .gradio-container {
    background: linear-gradient(135deg, #070a13 0%, #0f172a 100%) !important;
    color: #e5e7eb !important;
}
.gradio-container {
    max-width: 1300px !important;
    margin: 30px auto !important;
    padding: 0 20px !important;
}

/* 顶部标题栏样式 */
.header {
    text-align: center;
    margin-bottom: 35px;
    padding: 28px;
    background: rgba(30, 41, 59, 0.35);
    border: 1px solid rgba(251, 191, 36, 0.15);
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.header h1 {
    font-size: 3rem;
    font-weight: 800;
    color: #fbbf24;
    margin-bottom: 10px;
    letter-spacing: -0.025em;
    text-shadow: 0 2px 8px rgba(251, 191, 36, 0.2);
}
.header p {
    font-size: 1.2rem;
    color: #d1d5db;
    letter-spacing: 1px;
}

/* 卡片通用样式 */
.chatbot-wrap, .gr-box, .gr-panel, .gr-form, .chatbot, .textbox, .gr-markdown {
    border-radius: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s ease !important;
}

/* 输入框样式优化 */
textarea, input[type="text"] {
    background-color: #1e293b !important;
    color: #f9fafb !important;
    border: 1px solid #475569 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #fbbf24 !important;
    box-shadow: 0 0 0 2px rgba(251, 191, 36, 0.2) !important;
}

/* 按钮样式美化 */
button.primary {
    background: linear-gradient(135deg, #f59e0b, #fbbf24) !important;
    color: #0f172a !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    border: none !important;
}
button.secondary {
    background: rgba(71, 85, 105, 0.5) !important;
    color: #e5e7eb !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.2s ease !important;
}

/* 标题文本样式 */
h3 {
    color: #fbbf24 !important;
    font-weight: 600 !important;
    margin-bottom: 15px !important;
}

/* 聊天框滚动条美化 */
.chatbot::-webkit-scrollbar, .gr-markdown::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
.chatbot::-webkit-scrollbar-thumb, .gr-markdown::-webkit-scrollbar-thumb {
    background: rgba(251, 191, 36, 0.3);
    border-radius: 3px;
}
.chatbot::-webkit-scrollbar-track, .gr-markdown::-webkit-scrollbar-track {
    background: rgba(30, 41, 59, 0.2);
}
"""

# 构建Gradio页面主体（新版 Blocks 写法）
with gr.Blocks(title="ReAct AI Agent Framework Dashboard") as demo:
    # 顶部标题模块
    gr.HTML("""
        <div class="header">
            <h1>ReAct AI Agent Framework</h1>
            <p>基于 ReAct 范式的自主决策与工具调用闭环演示系统</p>
        </div>
    """)
    
    # 会话状态存储器
    history_state = gr.State([])
    
    # 工作目录配置行
    with gr.Row():
        workspace_input = gr.Textbox(
            label="📂 当前工作目录",
            value=os.path.abspath(os.path.dirname(__file__)),
            placeholder="请输入工作目录路径",
            scale=1
        )
        
    # 主体内容分区
    with gr.Row():
        # 左侧：对话交互区域
        with gr.Column(scale=3):
            gr.Markdown("### 💬 交互对话")
            # 移除 bubble_full_width，适配新版
            chatbot = gr.Chatbot(label="与 Agent 对话", height=520, show_label=False)
            msg = gr.Textbox(
                label="✍️ 输入您的问题",
                placeholder="例如：计算圆周率 3.14159 乘以半径 10 的平方，并将结果写入 area.txt 文件中",
                lines=2,
                max_lines=4
            )
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary", scale=2)
                clear_btn = gr.Button("清空历史", variant="secondary", scale=1)
                
        # 右侧：ReAct执行轨迹展示区
        with gr.Column(scale=2):
            gr.Markdown("### 📜 ReAct 运行轨迹")
            trace_view = gr.Markdown(value="等待提问以显示运行轨迹...", height=520)
            
    # ========== 事件绑定逻辑 ==========
    submit_event = submit_btn.click(
        predict, 
        inputs=[msg, history_state, workspace_input], 
        outputs=[chatbot, trace_view]
    )
    
    # 发送后清空输入框
    submit_event.then(lambda: "", inputs=None, outputs=msg)
    
    # 同步更新历史
    submit_event.then(lambda h: h, inputs=[chatbot], outputs=[history_state])

    # 回车提交
    submit_event_textbox = msg.submit(
        predict,
        inputs=[msg, history_state, workspace_input],
        outputs=[chatbot, trace_view]
    )
    submit_event_textbox.then(lambda: "", inputs=None, outputs=msg).then(
        lambda h: h, inputs=[chatbot], outputs=[history_state]
    )

    # 清空按钮
    def clear_all():
        return [], [], "等待提问以显示运行轨迹..."

    clear_btn.click(fn=clear_all, inputs=None, outputs=[chatbot, history_state, trace_view])

# 启动Web服务（新版 Gradio 只在这里加载 theme + css）
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        theme=theme, 
        css=css,
        inbrowser=True
    )
