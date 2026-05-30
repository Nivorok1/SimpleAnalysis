import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_dataset(size=1000, min_value=-10000, max_value=10000, seed=None, file_path="dataset.csv"):
    """Сгенерировать целочисленный набор данных и вернуть его как pandas Series."""
    rng = np.random.default_rng(seed)
    data = rng.integers(min_value, max_value + 1, size=size)
    dataset = pd.Series(data, name="values")
    dataset.to_csv(file_path, index=False)
    return dataset


def print_series_characteristics(series):
    """Рассчитать и вывести стандартные числовые характеристики pandas Series."""
    duplicated_count = series.duplicated(keep=False).sum()

    print("Стандартные числовые характеристики набора данных Series:")
    print(f"Минимальное значение: {series.min()}")
    print(f"Количество повторяющихся значений: {duplicated_count}")
    print(f"Максимальное значение: {series.max()}")
    print(f"Сумма чисел: {series.sum()}")
    print(f"Среднеквадратическое отклонение: {series.std()}")


def round_to_hundreds(series):
    """Округлить значения до сотен по математическому правилу."""
    values = series.to_numpy()
    rounded_values = np.sign(values) * np.floor(np.abs(values) / 100 + 0.5) * 100
    return pd.Series(rounded_values.astype(int), name="rounded_values")


def visualize_dataset(series, show=True):
    """Построить линейный график и гистограмму для набора данных Series."""
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, color="blue", label="Исходные значения")
    plt.title("Линейный график исходного набора данных")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.grid(True)
    plt.legend()
    plt.savefig("line_plot.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    rounded_series = round_to_hundreds(series)

    plt.figure(figsize=(12, 6))
    plt.hist(rounded_series, bins=30, color="green", edgecolor="black")
    plt.title("Гистограмма значений, округленных до сотен")
    plt.xlabel("Округленное значение")
    plt.ylabel("Количество")
    plt.grid(True)
    plt.savefig("histogram.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def create_dataframe_from_series(series):
    """Сформировать DataFrame из Series и добавить отсортированные столбцы."""
    sorted_ascending = series.sort_values(ascending=True).reset_index(drop=True)
    sorted_descending = series.sort_values(ascending=False).reset_index(drop=True)

    dataframe = pd.DataFrame({
        "Исходные значения": series.reset_index(drop=True),
        "Сортировка по возрастанию": sorted_ascending,
        "Сортировка по убыванию": sorted_descending,
    })

    return dataframe


def visualize_intermediate_analysis(dataframe, show=True):
    """Построить два линейных графика отсортированных значений на одном plt."""
    plt.figure(figsize=(12, 6))
    plt.plot(
        dataframe.index,
        dataframe["Сортировка по возрастанию"],
        color="blue",
        label="По возрастанию",
    )
    plt.plot(
        dataframe.index,
        dataframe["Сортировка по убыванию"],
        color="red",
        label="По убыванию",
    )
    plt.title("Графики отсортированных значений")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.grid(True)
    plt.legend()
    plt.savefig("sorted_values_plot.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    dataset = get_dataset()
    print(dataset)
    print_series_characteristics(dataset)
    visualize_dataset(dataset)
    dataframe = create_dataframe_from_series(dataset)
    print(dataframe)
    visualize_intermediate_analysis(dataframe)
