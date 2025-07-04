import gradio as gr

class DolorisPanel:
    def __init__(self):
        self.classification_type = None
        self.num_weeks = None
        self.selected_subjects = None
        self.algorithm = None

    def validate_and_submit(self, classification_type, num_weeks, selected_subjects, algorithm):
        errors = []

        # 校验分类类型
        if classification_type not in ['2分类', 'n分类']:
            errors.append("分类类型必须为 '2分类' 或 'n分类'。")

        # 校验周数
        if not isinstance(num_weeks, int) or num_weeks <= 0:
            errors.append("使用的周数必须为正整数。")

        # 校验选择的学科
        if not selected_subjects or not isinstance(selected_subjects, list):
            errors.append("必须至少选择一门学科。")

        # 校验算法
        if algorithm not in ['随机森林', 'SVM', 'KNN', '逻辑回归']:
            errors.append("请选择一个有效的算法。")

        if errors:
            return "\n".join(errors)

        # 参数通过校验，保存并打印
        self.classification_type = classification_type
        self.num_weeks = num_weeks
        self.selected_subjects = selected_subjects
        self.algorithm = algorithm

        print("分类类型:", self.classification_type)
        print("使用周数:", self.num_weeks)
        print("选择学科:", self.selected_subjects)
        print("使用算法:", self.algorithm)

        return "参数提交成功！请查看后端打印结果。"

    def launch(self):
        with gr.Blocks(title="Doloris 面板") as demo:
            gr.Markdown("## Doloris 参数配置面板")

            classification_type = gr.Radio(
                label="请选择分类类型",
                choices=["2分类", "n分类"],
                value="2分类"
            )

            num_weeks = gr.Number(
                label="使用几周的数据",
                value=4,
                precision=0,
                interactive=True
            )

            subject_choices = ["语文", "数学", "英语", "物理", "化学", "生物"]
            selected_subjects = gr.CheckboxGroup(
                label="选择使用哪几科的数据",
                choices=subject_choices
            )

            algorithm = gr.Radio(
                label="选择使用的算法",
                choices=["随机森林", "SVM", "KNN", "逻辑回归"],
                value="随机森林"
            )

            submit_btn = gr.Button("提交参数")

            output = gr.Textbox(label="结果输出", lines=3)

            submit_btn.click(
                fn=self.validate_and_submit,
                inputs=[classification_type, num_weeks, selected_subjects, algorithm],
                outputs=output
            )

        demo.launch()
        return