from pathlib import Path
import unittest
import warnings

import matplotlib.pyplot as plt
import numpy as np
from streamlit.testing.v1 import AppTest

from app import inverse_ranges, make_inverse_chart


APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class InverseCalculationTests(unittest.TestCase):
    def test_known_example_and_every_candidate(self):
        result = inverse_ranges(10_000, 5000, 60)
        expected_counts = tuple(k for k in range(5001) if f"{k / 5000 * 100:.1f}" == "60.0")
        self.assertEqual(result.target_counts, expected_counts)
        x, target, other = result.series(3000)
        for population, a, b in ((.3, 1, 2000 / 7000), (.5, .6, .4), (.8, .375, 1)):
            index = np.flatnonzero(x == population)[0]
            self.assertAlmostEqual(target[index], a)
            self.assertAlmostEqual(other[index], b)
        for k in result.target_counts:
            x, target, other = result.series(k)
            t = np.rint(x * 10_000).astype(int)
            self.assertTrue(np.all(t >= k))
            self.assertTrue(np.all(10_000 - t >= 5000 - k))
            np.testing.assert_allclose(target * t, k)
            np.testing.assert_allclose(other * (10_000 - t), 5000 - k)
        self.assertEqual(result.population_range, (expected_counts[0] / 10_000,
                                                  (5000 + expected_counts[-1]) / 10_000))

    def test_ranges_match_exhaustive_candidates(self):
        for m, percent in ((0, 63.2), (3, 33.3), (20, 60), (100, 50), (100, 0), (100, 100)):
            with self.subTest(m=m, percent=percent):
                result = inverse_ranges(100, m, percent)
                all_x, all_a, all_b = [], [], []
                for k in result.target_counts:
                    for t in range(k, 100 - (m - k) + 1):
                        all_x.append(t / 100)
                        if t:
                            all_a.append(k / t)
                        if t < 100:
                            all_b.append((m - k) / (100 - t))
                self.assertEqual(result.population_range, (min(all_x), max(all_x)))
                expected = tuple((min(v), max(v)) if v else None for v in (all_a, all_b))
                self.assertEqual(result.response_ranges(), expected)

    def test_rounding_including_halfway_cases_matches_existing_display(self):
        for percent in (0, 100, 12.2, 12.3, 63.2):
            expected = tuple(k for k in range(4001) if f"{k / 4000 * 100:.1f}" == f"{percent:.1f}")
            result = inverse_ranges(10_000, 4000, percent)
            self.assertEqual(result.target_counts, expected)
            self.assertGreater(len(expected), 1)

    def test_zero_respondents_ignores_percentage_and_avoids_division_warnings(self):
        result = inverse_ranges(100, 0, float("nan"))
        self.assertEqual(result.target_counts, (0,))
        self.assertEqual(result.population_range, (0, 1))
        self.assertEqual(result.response_ranges(), ((0, 0), (0, 0)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            x, a, b = result.series(0)
        self.assertTrue(np.isnan(a[0]))
        self.assertTrue(np.isnan(b[-1]))
        np.testing.assert_array_equal(a[1:], 0)
        np.testing.assert_array_equal(b[:-1], 0)

    def test_all_respondents_and_absent_groups(self):
        for percent in (0, 50, 100):
            with self.subTest(percent=percent):
                result = inverse_ranges(100, 100, percent)
                self.assertEqual(result.population_range, (percent / 100, percent / 100))
                expected = (None if percent == 0 else (1, 1),
                            None if percent == 100 else (1, 1))
                self.assertEqual(result.response_ranges(), expected)

    def test_invalid_inputs_and_inconsistent_percentage(self):
        for values in ((99, 1, 100), (101, 1, 100), (100_100, 1, 100),
                       (100, -1, 0), (100, 101, 50), (100, 1.5, 50),
                       (100, 3, 50), (100, 1, -1), (100, 1, 101),
                       (100, 1, float("nan"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                inverse_ranges(*values)

    def test_maximum_population_and_repeatability(self):
        result = inverse_ranges(100_000, 50_000, 50)
        self.assertEqual(result, inverse_ranges(100_000, 50_000, 50))
        self.assertGreater(len(result.target_counts), 1)
        for count in result.target_counts:
            x, a, b = result.series(count)
            self.assertEqual(len(x), 50_001)
            self.assertTrue(np.all((a >= 0) & (a <= 1)))
            self.assertTrue(np.all((b >= 0) & (b <= 1)))


class InverseChartTests(unittest.TestCase):
    def test_axis_series_legend_and_font(self):
        result = inverse_ranges(1000, 500, 60)
        fig = make_inverse_chart(result)
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_xlim(), (0, 100))
            self.assertEqual(ax.get_ylim(), (0, 100))
            self.assertEqual(ax.get_xlabel(), "母集団の対象者割合")
            self.assertEqual(ax.get_ylabel(), "回答率")
            self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                             ["対象者の回答率", "非対象者の回答率"])
            self.assertNotEqual(ax.lines[0].get_color(), ax.lines[1].get_color())
            np.testing.assert_allclose(ax.lines[0].get_xdata()[[0, -1]], [30, 80])
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                fig.canvas.draw()
        finally:
            plt.close(fig)

    def test_multiple_candidates_and_single_point_markers(self):
        for n, m, percent in ((1000, 1000, 100), (1000, 1000, 0),
                              (1000, 1000, 50), (10_000, 5000, 60)):
            with self.subTest(percent=percent):
                result = inverse_ranges(n, m, percent)
                fig = make_inverse_chart(result)
                try:
                    ax = fig.axes[0]
                    self.assertEqual(len(ax.get_legend().get_texts()), 2)
                    for line in ax.lines:
                        self.assertTrue(np.isfinite(line.get_ydata()).all())
                        if len(line.get_xdata()) == 1:
                            self.assertEqual(line.get_marker(), "o")
                        if len(result.target_counts) > 1:
                            self.assertLess(line.get_alpha(), 1)
                finally:
                    plt.close(fig)


class InverseAppTests(unittest.TestCase):
    def start_app(self):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def metrics(self, app):
        return {metric.label: metric.value for metric in app.tabs[1].get("metric")}

    def images(self, app):
        tab = app.tabs[1]
        return list(tab.get("image")) + list(tab.get("imgs"))

    def execute(self, app):
        app.button(key="inverse_run").click().run()
        self.assertEqual(len(app.exception), 0)

    def test_initial_inputs_and_tabs(self):
        app = self.start_app()
        self.assertEqual([tab.label for tab in app.tabs], ["シミュレーション", "取りうる値の逆算"])
        self.assertEqual(self.metrics(app), {})
        self.assertEqual(len(self.images(app)), 0)
        for key, expected in (("inverse_population", (10_000, 100, 100_000, 100)),
                              ("inverse_respondents", (3800, 0, 100_000, 1)),
                              ("inverse_percent", (63.2, 0, 100, .1))):
            widget = app.number_input(key=key)
            self.assertEqual((widget.value, widget.min, widget.max, widget.step), expected)

    def test_display_retention_rerun_and_rounding_note(self):
        app = self.start_app()
        self.execute(app)
        first = app.session_state["inverse_result"]
        values = self.metrics(app)
        self.assertEqual(len(values), 4)
        self.assertEqual(values["全体の回答率"], "38.0%")
        self.assertEqual(len(self.images(app)), 1)
        self.assertTrue(any("候補" in c.value for c in app.caption))
        app.run()
        self.assertEqual(app.session_state["inverse_result"], first)
        self.assertEqual(self.metrics(app), values)
        self.execute(app)
        self.assertIsNot(app.session_state["inverse_result"], first)

    def test_each_input_clears_results(self):
        app = self.start_app()
        for key, value in (("inverse_population", 20_000), ("inverse_respondents", 5000),
                           ("inverse_percent", 60.0)):
            self.execute(app)
            app.number_input(key=key).set_value(value).run()
            self.assertEqual(self.metrics(app), {})
            self.assertEqual(len(self.images(app)), 0)
            self.assertTrue(any("再表示" in info.value for info in app.info))
        self.execute(app)
        self.assertEqual(len(app.info), 0)

    def test_invalid_count_does_not_silently_change_input(self):
        app = self.start_app()
        self.execute(app)
        app.number_input(key="inverse_population").set_value(100).run()
        self.assertEqual(app.number_input(key="inverse_respondents").value, 3800)
        self.assertTrue(app.button(key="inverse_run").disabled)
        self.assertEqual(len(app.error), 1)
        self.assertEqual(self.metrics(app), {})

    def test_inconsistent_percentage_error_and_recovery(self):
        app = self.start_app()
        app.number_input(key="inverse_respondents").set_value(3)
        app.number_input(key="inverse_percent").set_value(50.0)
        self.execute(app)
        self.assertEqual(app.error[0].value, "回答者数と対象者割合が整合しません。")
        self.assertEqual(len(self.images(app)), 0)
        app.run()
        self.assertEqual(len(app.error), 1)
        app.number_input(key="inverse_percent").set_value(33.3).run()
        self.assertEqual(len(app.error), 0)
        self.execute(app)
        self.assertEqual(len(self.images(app)), 1)

    def test_zero_respondents_and_absent_groups(self):
        app = self.start_app()
        app.number_input(key="inverse_respondents").set_value(0).run()
        self.assertTrue(app.number_input(key="inverse_percent").disabled)
        self.execute(app)
        values = self.metrics(app)
        self.assertEqual(values["母集団の対象者割合"], "0.0% ～ 100.0%")
        self.assertEqual(values["対象者の回答率"], "0.0% ～ 0.0%")
        self.assertIn("絞り込めません", app.warning[0].value)
        self.assertEqual(len(self.images(app)), 1)
        app.number_input(key="inverse_population").set_value(100)
        app.number_input(key="inverse_respondents").set_value(100).run()
        self.assertFalse(app.number_input(key="inverse_percent").disabled)
        for percent, absent in ((0.0, "対象者の回答率"), (100.0, "非対象者の回答率")):
            app.number_input(key="inverse_percent").set_value(percent)
            self.execute(app)
            self.assertEqual(self.metrics(app)[absent], "算出不可")
            self.assertTrue(any("存在しない" in c.value for c in app.caption))

    def test_results_are_independent_between_tabs(self):
        app = self.start_app()
        app.button[0].click().run()
        simulation = app.session_state["result"]
        self.execute(app)
        self.assertIs(app.session_state["result"], simulation)
        inverse = app.session_state["inverse_result"]
        app.number_input(key="target_percent").set_value(50).run()
        self.assertEqual(len(app.tabs[0].get("metric")), 0)
        self.assertIs(app.session_state["inverse_result"], inverse)
        app.button[0].click().run()
        simulation = app.session_state["result"]
        app.number_input(key="inverse_percent").set_value(60.0).run()
        self.assertIs(app.session_state["result"], simulation)
        self.assertEqual(self.metrics(app), {})


if __name__ == "__main__":
    unittest.main()
