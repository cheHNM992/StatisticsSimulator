"""アンケートの回答率の違いによる構成の偏りをシミュレーションする。"""

from dataclasses import dataclass

import matplotlib

# Streamlit上で描画するため、デスクトップGUIを必要としないバックエンドを使う。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401 -- bundled Japanese font
import numpy as np
import streamlit as st
from matplotlib.ticker import PercentFormatter


@dataclass(frozen=True)
class SimulationResult:
    population_target_ratio: float
    respondent_target_ratio: float | None
    response_rate: float
    difference_pt: float | None
    respondent_count: int


def simulate(
    population_size: int,
    target_ratio: float,
    target_response_rate: float,
    other_response_rate: float,
    *,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """割合は0～1で渡す。rngの指定は再現可能なテスト用。"""
    if not 100 <= population_size <= 100_000 or population_size % 100:
        raise ValueError("母集団人数は100～100,000の100人刻みで指定してください。")
    if not all(0 <= value <= 1 for value in (
        target_ratio, target_response_rate, other_response_rate
    )):
        raise ValueError("割合と回答率は0～1で指定してください。")
    if rng is None:
        rng = np.random.default_rng()
    targets = rng.random(population_size) < target_ratio
    rates = np.where(targets, target_response_rate, other_response_rate)
    respondents = rng.random(population_size) < rates
    count = int(np.count_nonzero(respondents))
    population_ratio = float(np.mean(targets))
    observed_ratio = float(np.mean(targets[respondents])) if count else None
    return SimulationResult(
        population_target_ratio=population_ratio,
        respondent_target_ratio=observed_ratio,
        response_rate=count / population_size,
        difference_pt=(observed_ratio - population_ratio) * 100
        if observed_ratio is not None else None,
        respondent_count=count,
    )


def make_comparison_chart(result: SimulationResult):
    """回答者がいる場合の比較グラフを生成する。"""
    if result.respondent_target_ratio is None:
        raise ValueError("回答者が0人のため、比較グラフを作成できません。")
    fig, ax = plt.subplots(figsize=(6, 4))
    values = [result.population_target_ratio * 100, result.respondent_target_ratio * 100]
    bars = ax.bar(["母集団", "回答者"], values, color=["#4878a8", "#e58c46"])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylabel("対象者割合")
    for bar, value in zip(bars, values):
        # 100%の棒でも値が描画領域からはみ出さないようにする。
        inside = value >= 90
        ax.annotate(
            f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, -6 if inside else 4), textcoords="offset points",
            ha="center", va="top" if inside else "bottom",
            color="white" if inside else "black",
        )
    fig.tight_layout()
    return fig


def clear_result():
    st.session_state.pop("result", None)
    st.session_state["inputs_changed"] = True


@dataclass(frozen=True)
class InverseResult:
    population_size: int
    respondent_count: int
    target_counts: tuple[int, ...]

    @property
    def population_range(self):
        return (self.target_counts[0] / self.population_size,
                (self.population_size - self.respondent_count + self.target_counts[-1])
                / self.population_size)

    def response_ranges(self):
        target_bounds, other_bounds = [], []
        for count in self.target_counts:
            other = self.respondent_count - count
            lower, upper = count, self.population_size - other
            if upper > 0:
                target_bounds.extend((count / upper, count / max(lower, 1)))
            if lower < self.population_size:
                other_bounds.extend((other / (self.population_size - lower),
                                     other / (self.population_size - min(upper, self.population_size - 1))))
        return tuple((min(values), max(values)) if values else None
                     for values in (target_bounds, other_bounds))

    def series(self, target_count):
        other = self.respondent_count - target_count
        counts = np.arange(target_count, self.population_size - other + 1)
        target_rates = np.full(counts.size, np.nan)
        other_rates = np.full(counts.size, np.nan)
        np.divide(target_count, counts, out=target_rates, where=counts > 0)
        np.divide(other, self.population_size - counts, out=other_rates,
                  where=counts < self.population_size)
        return counts / self.population_size, target_rates, other_rates


def inverse_ranges(population_size: int, respondent_count: int,
                   respondent_target_percent: float) -> InverseResult:
    """表示割合と一致する整数人数を列挙し、実現可能な構成を逆算する。"""
    if (not isinstance(population_size, (int, np.integer))
            or not 100 <= population_size <= 100_000 or population_size % 100):
        raise ValueError("母集団人数は100～100,000の100人刻みで指定してください。")
    if (not isinstance(respondent_count, (int, np.integer))
            or not 0 <= respondent_count <= population_size):
        raise ValueError("回答者数は0以上、母集団人数以下で指定してください。")
    if respondent_count == 0:
        return InverseResult(population_size, 0, (0,))
    if not 0 <= respondent_target_percent <= 100:
        raise ValueError("回答者に占める対象者割合は0～100%で指定してください。")
    display_value = f"{respondent_target_percent:.1f}"
    # NumPyのroundではなく、既存結果表示と同じPythonの書式化で照合する。
    candidates = tuple(count for count in range(respondent_count + 1)
                       if f"{count / respondent_count * 100:.1f}" == display_value)
    if not candidates:
        raise ValueError("回答者数と対象者割合が整合しません。")
    return InverseResult(population_size, respondent_count, candidates)


def make_inverse_chart(result: InverseResult):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ("#4878a8", "#e58c46")
    labels = ("対象者の回答率", "非対象者の回答率")
    alpha = 0.25 if len(result.target_counts) > 1 else 1
    labeled = [False, False]
    for count in result.target_counts:
        population, *rates = result.series(count)
        for index, rate in enumerate(rates):
            valid = np.isfinite(rate)
            if not valid.any():
                continue
            ax.plot(population[valid] * 100, rate[valid] * 100,
                    color=colors[index], alpha=alpha,
                    marker="o" if valid.sum() == 1 else None,
                    label=labels[index] if not labeled[index] else "_nolegend_")
            labeled[index] = True
    for index, present in enumerate(labeled):
        if not present:
            ax.plot([], [], color=colors[index], label=labels[index])
    ax.set(xlim=(0, 100), ylim=(0, 100),
           xlabel="母集団の対象者割合", ylabel="回答率")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend(loc="upper center")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def clear_inverse_result():
    st.session_state.pop("inverse_result", None)
    st.session_state.pop("inverse_error", None)
    st.session_state["inverse_inputs_changed"] = True


def render_simulation():
    columns = st.columns(2)
    with columns[0]:
        population_size = st.number_input(
            "母集団人数", min_value=100, max_value=100_000, value=10_000,
            step=100, key="population_size", on_change=clear_result,
        )
        target_percent = st.number_input(
            "対象者の割合（設定値） [%]", min_value=0, max_value=100, value=30,
            step=1, key="target_percent", on_change=clear_result,
        )
    with columns[1]:
        target_response = st.number_input(
            "対象者の回答率 [%]", min_value=0, max_value=100, value=80,
            step=1, key="target_response", on_change=clear_result,
        )
        other_response = st.number_input(
            "非対象者の回答率 [%]", min_value=0, max_value=100, value=20,
            step=1, key="other_response", on_change=clear_result,
        )
    if st.button("シミュレーション実行"):
        st.session_state["result"] = simulate(
            population_size, target_percent / 100,
            target_response / 100, other_response / 100,
        )
        st.session_state["inputs_changed"] = False

    result = st.session_state.get("result")
    if result is None:
        if st.session_state.get("inputs_changed", False):
            st.info("入力が変更されました。シミュレーションを再実行してください。")
        return

    st.subheader("結果")
    if result.respondent_count == 0:
        st.warning("回答者が0人のため、観測値を計算できません。")
    columns = st.columns(2)
    columns[0].metric("母集団の対象者割合", f"{result.population_target_ratio * 100:.1f}%")
    columns[1].metric("回答者に占める対象者割合", "算出不可"
                      if result.respondent_target_ratio is None
                      else f"{result.respondent_target_ratio * 100:.1f}%")
    columns[0].metric("全体の回答率", f"{result.response_rate * 100:.1f}%")
    columns[1].metric("母集団との差", "算出不可"
                      if result.difference_pt is None else f"{result.difference_pt:.1f} pt")
    st.metric("回答者数", f"{result.respondent_count:,}人")
    if result.respondent_count:
        fig = make_comparison_chart(result)
        try:
            st.pyplot(fig)
        finally:
            plt.close(fig)


def format_range(bounds):
    if bounds is None:
        return "算出不可"
    return f"{bounds[0] * 100:.1f}% ～ {bounds[1] * 100:.1f}%"


def render_inverse():
    population_size = st.number_input(
        "母集団人数", min_value=100, max_value=100_000, value=10_000, step=100,
        key="inverse_population", on_change=clear_inverse_result,
    )
    respondent_count = st.number_input(
        "回答者数", min_value=0, max_value=100_000, value=3800, step=1,
        key="inverse_respondents", on_change=clear_inverse_result,
    )
    percent = st.number_input(
        "回答者に占める対象者割合 [%]", min_value=0.0, max_value=100.0,
        value=63.2, step=0.1, format="%.1f", disabled=respondent_count == 0,
        key="inverse_percent", on_change=clear_inverse_result,
    )
    invalid_count = respondent_count > population_size
    if invalid_count:
        st.error("回答者数は母集団人数以下で指定してください。")
    if st.button("取りうる値を表示", key="inverse_run", disabled=invalid_count):
        try:
            st.session_state["inverse_result"] = inverse_ranges(
                population_size, respondent_count, percent,
            )
            st.session_state.pop("inverse_error", None)
        except ValueError as error:
            st.session_state.pop("inverse_result", None)
            st.session_state["inverse_error"] = str(error)
        st.session_state["inverse_inputs_changed"] = False
    if st.session_state.get("inverse_error"):
        st.error(st.session_state["inverse_error"])
    result = st.session_state.get("inverse_result")
    if result is None:
        if st.session_state.get("inverse_inputs_changed", False):
            st.info("入力が変更されました。取りうる値を再表示してください。")
        return
    st.subheader("取りうる値")
    st.metric("全体の回答率", f"{result.respondent_count / result.population_size * 100:.1f}%")
    target_range, other_range = result.response_ranges()
    for column, label, bounds in zip(st.columns(3),
                                    ("母集団の対象者割合", "対象者の回答率", "非対象者の回答率"),
                                    (result.population_range, target_range, other_range)):
        column.metric(label, format_range(bounds))
    if result.respondent_count == 0:
        st.warning("回答者が0人のため、母集団の対象者割合を絞り込めません。")
    if len(result.target_counts) > 1:
        st.caption(f"表示割合の丸めにより、対象者の回答人数は"
                   f"{result.target_counts[0]:,}～{result.target_counts[-1]:,}人"
                   f"（{len(result.target_counts)}候補）です。")
    if result.target_counts[0] == 0:
        st.caption("母集団の対象者割合が0%の候補では、対象者が存在しないため、対象者の回答率は算出不可です。")
    if result.target_counts[-1] == result.respondent_count:
        st.caption("母集団の対象者割合が100%の候補では、非対象者が存在しないため、非対象者の回答率は算出不可です。")
    fig = make_inverse_chart(result)
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)
    st.caption("3項目の値は互いに関係します。範囲内の値を自由に組み合わせられるわけではありません。"
               "同じ母集団割合・同じ回答人数の候補に対応する回答率の組み合わせが成立します。")


def main():
    st.set_page_config(page_title="アンケート・バイアスシミュレータ")
    st.title("アンケート・バイアスシミュレータ")
    simulation_tab, inverse_tab = st.tabs(["シミュレーション", "取りうる値の逆算"])
    with simulation_tab:
        render_simulation()
    with inverse_tab:
        render_inverse()


if __name__ == "__main__":
    main()
