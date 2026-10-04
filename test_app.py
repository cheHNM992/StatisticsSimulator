import itertools
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import matplotlib.pyplot as plt
import numpy as np
from streamlit.testing.v1 import AppTest

from app import make_comparison_chart, simulate


APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class SimulationTests(unittest.TestCase):
    def test_calculation_matches_individual_draws(self):
        rng = np.random.default_rng(42)
        targets = rng.random(1000) < 0.3
        respondents = rng.random(1000) < np.where(targets, 0.8, 0.2)
        result = simulate(1000, 0.3, 0.8, 0.2, rng=np.random.default_rng(42))
        self.assertEqual(result.respondent_count, int(respondents.sum()))
        self.assertEqual(result.population_target_ratio, float(targets.mean()))
        self.assertEqual(result.respondent_target_ratio, float(targets[respondents].mean()))
        self.assertEqual(result.response_rate, respondents.mean())
        self.assertEqual(result.difference_pt,
                         (targets[respondents].mean() - targets.mean()) * 100)

    def test_default_rng_is_used_without_injected_generator(self):
        with patch("app.np.random.default_rng", return_value=np.random.default_rng(7)) as factory:
            simulate(100, 0.3, 0.8, 0.2)
        factory.assert_called_once_with()

    def test_everyone_responds(self):
        result = simulate(10_000, 0.3, 1, 1, rng=np.random.default_rng(1))
        self.assertEqual(result.population_target_ratio, result.respondent_target_ratio)
        self.assertEqual(result.difference_pt, 0)
        self.assertEqual(result.response_rate, 1)
        self.assertEqual(result.respondent_count, 10_000)

    def test_no_respondents(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = simulate(100, 0.3, 0, 0, rng=np.random.default_rng(1))
        self.assertIsNone(result.respondent_target_ratio)
        self.assertIsNone(result.difference_pt)
        self.assertEqual(result.respondent_count, 0)
        self.assertEqual(result.response_rate, 0)
        self.assertTrue(0 <= result.population_target_ratio <= 1)

    def test_all_percentage_endpoints_and_population_limits(self):
        for size, target, response, other in itertools.product((100, 100_000), (0, 1), (0, 1), (0, 1)):
            with self.subTest(size=size, target=target, response=response, other=other):
                result = simulate(size, target, response, other, rng=np.random.default_rng(2))
                expected_rate = response if target else other
                self.assertEqual(result.population_target_ratio, target)
                self.assertEqual(result.response_rate, expected_rate)
                self.assertEqual(result.respondent_count, size * expected_rate)
                if expected_rate:
                    self.assertEqual(result.respondent_target_ratio, target)
                    self.assertEqual(result.difference_pt, 0)
                else:
                    self.assertIsNone(result.respondent_target_ratio)

    def test_equal_response_rates_give_similar_ratios(self):
        result = simulate(100_000, 0.3, 0.5, 0.5, rng=np.random.default_rng(3))
        self.assertLess(abs(result.difference_pt), 1)

    def test_higher_target_response_rate_increases_observed_ratio(self):
        equal = simulate(100_000, 0.3, 0.2, 0.2, rng=np.random.default_rng(4))
        biased = simulate(100_000, 0.3, 0.8, 0.2, rng=np.random.default_rng(4))
        self.assertEqual(equal.population_target_ratio, biased.population_target_ratio)
        self.assertGreater(biased.respondent_target_ratio, equal.respondent_target_ratio)
        self.assertGreater(biased.difference_pt, 25)

    def test_invalid_inputs(self):
        for args in ((0, .3, .8, .2), (101, .3, .8, .2),
                     (100_100, .3, .8, .2), (100, -.1, .8, .2),
                     (100, .3, 1.1, .2), (100, .3, .8, float("nan"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                simulate(*args)


class ChartTests(unittest.TestCase):
    def test_chart_values_axis_labels_and_japanese_font(self):
        for target in (0, .3, 1):
            with self.subTest(target=target):
                result = simulate(1000, target, 1, 1, rng=np.random.default_rng(5))
                fig = make_comparison_chart(result)
                try:
                    ax = fig.axes[0]
                    self.assertEqual(ax.get_ylim(), (0, 100))
                    self.assertEqual([bar.get_height() for bar in ax.patches],
                                     [result.population_target_ratio * 100,
                                      result.respondent_target_ratio * 100])
                    self.assertEqual([label.get_text() for label in ax.get_xticklabels()],
                                     ["母集団", "回答者"])
                    self.assertEqual([text.get_text() for text in ax.texts],
                                     [f"{result.population_target_ratio * 100:.1f}%"] * 2)
                    # 実際の描画で日本語グリフ欠落などの警告がないことを確認。
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", UserWarning)
                        fig.canvas.draw()
                finally:
                    plt.close(fig)

    def test_zero_respondents_cannot_be_compared(self):
        result = simulate(100, .3, 0, 0)
        with self.assertRaises(ValueError):
            make_comparison_chart(result)


class AppTests(unittest.TestCase):
    def start_app(self):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def metrics(self, app):
        return {metric.label: metric.value for metric in app.metric}

    def charts(self, app):
        # Streamlitのバージョンにより画像要素の名前が異なる。
        return list(app.get("image")) + list(app.get("imgs"))

    def test_initial_screen_and_input_configuration(self):
        app = self.start_app()
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(self.charts(app)), 0)
        expected = {
            "population_size": (10_000, 100, 100_000, 100),
            "target_percent": (30, 0, 100, 1),
            "target_response": (80, 0, 100, 1),
            "other_response": (20, 0, 100, 1),
        }
        for key, (value, minimum, maximum, step) in expected.items():
            widget = app.number_input(key=key)
            self.assertEqual(widget.value, value)
            self.assertEqual((widget.min, widget.max, widget.step), (minimum, maximum, step))
        self.assertEqual(app.button[0].label, "シミュレーション実行")

    def test_run_preserve_result_and_run_again(self):
        app = self.start_app()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 5)
        self.assertEqual(len(self.charts(app)), 1)
        first_result = app.session_state["result"]
        first_metrics = self.metrics(app)
        app.run()
        self.assertEqual(app.session_state["result"], first_result)
        self.assertEqual(self.metrics(app), first_metrics)
        # オブジェクトを作り直すことを確認し、偶然同じ乱数結果でも失敗しない。
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIsNot(app.session_state["result"], first_result)

    def test_changing_each_input_clears_results_and_guides_rerun(self):
        app = self.start_app()
        for key, value in (("population_size", 100), ("target_percent", 0),
                           ("target_response", 100), ("other_response", 100)):
            with self.subTest(key=key):
                app.button[0].click().run()
                app.number_input(key=key).set_value(value).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.metric), 0)
                self.assertEqual(len(self.charts(app)), 0)
                self.assertIn("再実行", app.info[0].value)
                app.run()
                self.assertEqual(len(app.metric), 0)
                app.button[0].click().run()
                self.assertEqual(len(app.metric), 5)
                self.assertEqual(len(app.info), 0)

    def test_zero_respondents_screen(self):
        app = self.start_app()
        app.number_input(key="target_response").set_value(0)
        app.number_input(key="other_response").set_value(0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.warning[0].value, "回答者が0人のため、観測値を計算できません。")
        values = self.metrics(app)
        self.assertEqual(values["回答者に占める対象者割合"], "算出不可")
        self.assertEqual(values["母集団との差"], "算出不可")
        self.assertEqual(values["全体の回答率"], "0.0%")
        self.assertEqual(values["回答者数"], "0人")
        self.assertRegex(values["母集団の対象者割合"], r"^\d+\.\d%$")
        self.assertEqual(len(self.charts(app)), 0)

    def test_all_percentage_endpoints_on_screen(self):
        app = self.start_app()
        app.number_input(key="population_size").set_value(100)
        for target, response, other in itertools.product((0, 100), repeat=3):
            with self.subTest(target=target, response=response, other=other):
                app.number_input(key="target_percent").set_value(target)
                app.number_input(key="target_response").set_value(response)
                app.number_input(key="other_response").set_value(other)
                app.button[0].click().run()
                self.assertEqual(len(app.exception), 0)
                values = self.metrics(app)
                expected_rate = response if target else other
                self.assertEqual(values["母集団の対象者割合"], f"{target:.1f}%")
                self.assertEqual(values["全体の回答率"], f"{expected_rate:.1f}%")
                self.assertEqual(values["母集団との差"], "0.0 pt" if expected_rate else "算出不可")
                self.assertEqual(len(self.charts(app)), 1 if expected_rate else 0)


if __name__ == "__main__":
    unittest.main()
