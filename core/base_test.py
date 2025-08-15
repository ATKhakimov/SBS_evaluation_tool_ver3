from abc import ABC, abstractmethod
from typing import List, Dict, Union
import itertools

# Базовый абстрактный класс теста
class BaseModelTest(ABC):
    def __init__(self, test_code: str, test_name: str):
        self.test_code = test_code
        self.test_name = test_name
        self.test_signal = None  # string: "green", "yellow", "red"
        self.test_meta = {}      # dict with metadata: summary values, counts, etc.

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, str]:
        """
        Выполнить тест и вернуть словарь вида {feature_name: html_block, ..., 'DASHBOARD': dashboard_html (опционально)}
        """
        pass

class TestGroupBuilder:
    def __init__(self, group_code: str, group_name: str):
        self.group_code = group_code
        self.group_name = group_name
        self.tests: List[BaseModelTest] = []
        self.test_results: Dict[str, Dict[str, str]] = {}


    def add_test(self, test: BaseModelTest):
        self.tests.append(test)

    def run_all_tests(self, test_params: Dict[str, dict]):
        for test in self.tests:
            params = test_params.get(test.test_code, {})
            self.test_results[test.test_code] = test.run(**params)
    
    def generate_group_menu(self) -> str:
        signal_colors = {
            "green": "#e6f4ea",   # light green
            "orange": "#fff3cd",  # light orange instead of yellow
            "yellow": "#fff3cd",  # light yellow
            "red": "#fbeaea"      # light red
        }   

        return ''.join([
            f'<li onclick="showTest(\'{self.group_code}_{test.test_code}\')" '
            f'style="background-color: {signal_colors.get(test.test_signal, "#ddd")};">'
            f'{test.test_code} {test.test_name}</li>'
            for test in self.tests
        ])

    def generate_group_content(self) -> str:
        html_blocks = ""

        # --- Signal Summary Table ---
        signal_summary = "<h3>Test Signal Summary</h3><table border='1' cellpadding='5'><tr><th>Test</th><th>Signal</th></tr>"
        for test in self.tests:
            signal = getattr(test, "test_signal", None)
            color = {"green": "green", "yellow": "yellow", "red": "red"}.get(signal, "black")
            signal_display = signal.upper() if signal else "-"
            signal_summary += f"<tr><td>{test.test_code} {test.test_name}</td><td style='color:{color}'><b>{signal_display}</b></td></tr>"
        signal_summary += "</table><hr>"

        html_blocks += f"""
        <div id="signal_summary_{self.group_code}" class="test-block" style="display:none">
            {signal_summary}
        </div>
        """

        # --- Per-test blocks ---
        for test in self.tests:
            test_id = f"{self.group_code}_{test.test_code}"
            features = self.test_results[test.test_code]
            feature_blocks = ""

            if "DASHBOARD" in features:
                feature_blocks += f"<h3> Dashboard</h3>{features['DASHBOARD']}<hr>"

            for feat, html in features.items():
                if feat != "DASHBOARD":
                    feature_blocks += f"""
                    <details>
                        <summary><b>{feat}</b></summary>
                        {html}
                    </details>
                    <br>
                    """

            html_blocks += f"""
            <div id="{test_id}" class="test-block" style="display:none">
                <h2>{test.test_code} {test.test_name}</h2>
                {feature_blocks}
            </div>
            """

        return html_blocks


class ModelReportBuilder:
    def __init__(self):
        self.groups: List[TestGroupBuilder] = []

    def add_group(self, group: TestGroupBuilder):
        self.groups.append(group)

    def generate_html(self) -> str:
        tab_headers = ''.join([
            f'<button class="tablink" onclick="openGroup(event, \'group_{group.group_code}\')">{group.group_name}</button>'
            for group in self.groups
        ])

        group_tabs = ''
        for group in self.groups:
            menu = group.generate_group_menu()
            content = group.generate_group_content()
            group_tabs += f'''
            <div id="group_{group.group_code}" class="tabcontent" style="display:none;">
                <ul class="test-list">{menu}</ul>
                {content}
            </div>
            '''

        return f"""
        <html><head>
        <style>
            .tablink {{
                background-color: #ddd;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 10px 15px;
                font-size: 16px;
            }}
            .tabcontent {{
                display: none;
                padding: 20px;
            }}
            .test-list {{
                list-style-type: none;
                padding: 0;
                margin: 0 0 20px 0;
                background-color: #f0f0f0;
                overflow-x: auto;
                white-space: nowrap;
            }}
            .test-list li {{
                display: inline-block;
                padding: 8px 15px;
                cursor: pointer;
                background-color: #ddd;
                margin-right: 5px;
                border-radius: 4px;
            }}
            .test-block {{
                margin-top: 20px;
            }}
        </style>
        </head><body>

        <h1> <strong style="color: green;">SBS</strong> Model Evaluation Tool </h1>
        {tab_headers}
        {group_tabs}

        <script>
        function openGroup(evt, groupId) {{
            let i, content, tabs;
            content = document.getElementsByClassName("tabcontent");
            for (i = 0; i < content.length; i++) {{
                content[i].style.display = "none";
            }}
            tabs = document.getElementsByClassName("tablink");
            for (i = 0; i < tabs.length; i++) {{
                tabs[i].style.backgroundColor = "";
            }}
            document.getElementById(groupId).style.display = "block";
            evt.currentTarget.style.backgroundColor = "#ccc";

            // Auto-open signal summary when group opens
            let groupIdParts = groupId.split("_");
            let summaryBlock = document.getElementById("signal_summary_" + groupIdParts[1]);
            if (summaryBlock) {{
                let testBlocks = document.getElementsByClassName("test-block");
                for (let i = 0; i < testBlocks.length; i++) {{
                    testBlocks[i].style.display = "none";
                }}
                summaryBlock.style.display = "block";
            }}
        }}

        function showTest(testId) {{
            let blocks = document.getElementsByClassName("test-block");
            for (let i = 0; i < blocks.length; i++) {{
                blocks[i].style.display = "none";
            }}
            let block = document.getElementById(testId);
            if (block) {{
                block.style.display = "block";
            }}
        }}

        document.getElementsByClassName("tablink")[0].click();
        </script>
        </body></html>
        """
