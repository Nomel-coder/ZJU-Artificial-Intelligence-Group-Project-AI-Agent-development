import os
import gradio as gr
from agent import ReActAgent

# Initialize agent
agent = ReActAgent()

def predict(message, history, workspace_dir):
    if not message.strip():
        yield history, "请输入有效的问题。"
        return
        
    # Dynamically update the working directory env var based on UI input
    if workspace_dir and workspace_dir.strip():
        os.environ["AGENT_WORKSPACE_DIR"] = os.path.abspath(workspace_dir.strip())
        
    agent.reset()
    trace_log = ""
    
    # Initialize history list using dict format for Gradio 6.0+ chatbot
    chat_history = list(history) if history else []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": "思考中..."})
    
    yield chat_history, "正在启动 ReAct 推理闭环..."
    
    steps = agent.run(message)
    bot_response = ""
    
    for step in steps:
        step_type = step["type"]
        if step_type == "thought":
            trace_log += f"### Thought\n{step['content']}\n\n"
        elif step_type == "tool_call":
            trace_log += f"### Call Tool: `{step['tool']}`\n**Input:** `{step['input']}`\n\n"
        elif step_type == "observation":
            trace_log += f"### Observation\n```text\n{step['content']}\n```\n\n"
        elif step_type == "final_answer":
            bot_response = step["content"]
            trace_log += f"### Final Answer\n{bot_response}\n"
        elif step_type == "error":
            trace_log += f"### Error\n{step['content']}\n\n"
            
        current_bot_text = bot_response if bot_response else "正在执行步骤，请看右侧执行轨迹..."
        chat_history[-1] = {"role": "assistant", "content": current_bot_text}
        yield chat_history, trace_log
        
    if not bot_response:
        chat_history[-1] = {"role": "assistant", "content": "未能得出最终结论，请查看运行轨迹。"}
        yield chat_history, trace_log

# Custom Theme for a premium Slate Dark Mode look
theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

css = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #e5e7eb !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 40px auto !important;
}
.header {
    text-align: center;
    margin-bottom: 30px;
    padding: 24px;
    background: rgba(17, 24, 39, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    backdrop-filter: blur(10px);
}
.header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: #f59e0b;
    margin-bottom: 8px;
    letter-spacing: -0.025em;
}
.header p {
    font-size: 1.15rem;
    color: #9ca3af;
}
/* Premium glassmorphism effects on cards */
.chatbot-wrap, .gr-box, .gr-panel, .gr-form, .chatbot, .textbox {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    background: rgba(17, 24, 39, 0.5) !important;
    backdrop-filter: blur(10px);
}
/* Ensure dark mode input styling */
textarea, input[type="text"] {
    background-color: #1f2937 !important;
    color: #ffffff !important;
    border: 1px solid #374151 !important;
}
"""

with gr.Blocks(title="ReAct AI Agent Framework Dashboard") as demo:
    gr.HTML(
        """
        <div class="header">
            <h1>ReAct AI Agent Framework</h1>
            <p>基于 ReAct 范式的自主决策与工具调用闭环演示系统</p>
        </div>
        """
    )
    
    # Use gr.State to maintain chat history state
    history_state = gr.State([])
    
    # Global settings
    with gr.Row():
        workspace_input = gr.Textbox(
            label="当前工作目录 (Working Directory)",
            value=os.path.abspath(os.path.dirname(__file__)),
            placeholder="请输入工作目录路径，Agent 的文件读写操作将在此目录下进行",
            scale=1
        )
        
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 交互对话")
            chatbot = gr.Chatbot(label="与 Agent 对话", height=500, show_label=False)
            msg = gr.Textbox(
                label="输入您的问题",
                placeholder="例如：计算圆周率 3.14159 乘以半径 10 的平方，并将结果写入 area.txt 文件中",
                lines=2,
                max_lines=4
            )
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空历史")
                
        with gr.Column(scale=2):
            gr.Markdown("### ReAct 运行轨迹 (Reasoning & Tools Trace)")
            trace_view = gr.Markdown(
                value="等待提问以显示运行轨迹..."
            )
            
    # Event bindings
    submit_event = submit_btn.click(
        predict, 
        inputs=[msg, history_state, workspace_input], 
        outputs=[chatbot, trace_view]
    )
    
    # Clear the input textbox
    submit_event.then(
        fn=lambda: "",
        inputs=None,
        outputs=msg
    )
    
    # Update the history state when prediction completes
    submit_event.then(
        fn=lambda chat_history: chat_history,
        inputs=[chatbot],
        outputs=[history_state]
    )
    
    # Enter key behavior on Textbox
    submit_event_textbox = msg.submit(
        predict,
        inputs=[msg, history_state, workspace_input],
        outputs=[chatbot, trace_view]
    )
    submit_event_textbox.then(
        fn=lambda: "",
        inputs=None,
        outputs=msg
    ).then(
        fn=lambda chat_history: chat_history,
        inputs=[chatbot],
        outputs=[history_state]
    )

    # Clear button action
    def clear_all():
        return [], [], "等待提问以显示运行轨迹..."

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[chatbot, history_state, trace_view]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        theme=theme, 
        css=css
    )
