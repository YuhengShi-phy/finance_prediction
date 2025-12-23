from sklearn.utils import class_weight
import plotter as pt
import model as md
import data_preprocess as dp
import evaluation as eval
import pandas as pd
import numpy as np

# ----------
# Next step:
# these features may be divided into several parts, including
# price, volume, technical, and volatility
# But it might take a fucking lot of time, for god's sake!

feature_columns = [
    "n_close",
    "amount_delta",
    "n_midprice",
    "n_bid1",
    "n_bsize1",
    "n_bid2",
    "n_bsize2",
    "n_bid3",
    "n_bsize3",
    "n_bid4",
    "n_bsize4",
    "n_bid5",
    "n_bsize5",
    "n_ask1",
    "n_asize1",
    "n_ask2",
    "n_asize2",
    "n_ask3",
    "n_asize3",
    "n_ask4",
    "n_asize4",
    "n_ask5",
    "n_asize5",
    "bid_ask_spread",
    # "size_imbalance_1",
    # "size_imbalance_2",
    # "size_imbalance_3",
    # "size_imbalance_4",
    # "size_imbalance_5",
    # "weighted_midprice",
    # "microprice",
    "bid_depth",
    "ask_depth",
    "total_depth",
    "depth_imbalance",
    "bid_slope_2",
    "ask_slope_2",
    "bid_slope_3",
    "ask_slope_3",
    "bid_slope_4",
    "ask_slope_4",
    "bid_slope_5",
    "ask_slope_5",
    "buy_pressure",
    "sell_pressure",
    "time_sin",
    "time_cos",
]


def main():
    print("-" * 50)
    sequence_length = 50
    time_delay = 5
    # df_with_features = pd.read_csv("./df_with_features.csv")
    # df_with_features[f"return_after_{time_delay}"] = (
    #     df_with_features["n_midprice"].shift(-time_delay)
    #     - df_with_features["n_midprice"]
    # ) / (1 + df_with_features["n_midprice"])
    # df_with_features = df_with_features.tail(len(df_with_features) - 51)
    # df_with_features = df_with_features.head(len(df_with_features) - 10)

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    for i in range(10):

        df_raw = pd.read_csv(f"./merged_data/merged_{i}.csv")
        df_with_features = dp.create_all_features(df_raw)
        df_with_features = df_with_features.tail(len(df_with_features) - 51)
        df_with_features = df_with_features.head(len(df_with_features) - 10)

        sequence_length = 50
        time_delay = 5
        X_single, y_single = dp.sequentialize_certain_features(
            df_with_features,
            dp.selected_features,
            f"label_{time_delay}",
            sequence_length,
        )
        (X_train_single, X_test_single, y_train_single, y_test_single) = (
            dp.split_and_scale(X_single, y_single, test_size=0.2)
        )
        X_train_list.append(X_train_single)
        X_test_list.append(X_test_single)
        y_train_list.append(y_train_single)
        y_test_list.append(y_test_single)

    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, {y_test.shape}")

    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = md.build_classification_model(input_shape)
    model.summary()

    # 训练模型
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=128,
        verbose=1,
        class_weight={0: 4, 1: 1, 2: 3},
    )

    # 预测示例
    y_pred = model.predict(X_test)
    # pt.plot_predict_curve(y_test, y_pred)
    y_pred = np.argmax(y_pred, axis=1)

    # X_test_original = price_scaler.inverse_transform(X_test[:, 99, 0:3].reshape(-1, 3))

    # y_pred = eval.get_label(y_pred, X_test_original[:, 1], 5)
    # y_test = eval.get_label(y_test, X_test_original[:, 1], 5)
    # print(f"The first 20 pred labels: {y_pred[:20]}")
    # print(f"The first 20 true labels: {y_test[:20]}")

    test_score = eval.calculate_f_beta_multiclass(y_test, y_pred)
    # test_pnl_average = eval.calculate_pnl_average(df_with_features, y_pred, time_delay)
    print(f"The f beta score on test: {test_score}")
    # print(f"The pnl average on test: {test_pnl_average}")

    y_train_pred = model.predict(X_train)
    # pt.plot_predict_curve(y_train, y_train_pred)
    y_train_pred = np.argmax(y_train_pred, axis=1)
    # X_train_original = price_scaler.inverse_transform(
    #     X_train[:, 99, 0:3].reshape(-1, 3)
    # )

    # y_train_pred = eval.get_label(y_train_pred, X_train_original[:, 1], 5)
    # y_train = eval.get_label(y_train, X_train_original[:, 1], 5)

    train_score = eval.calculate_f_beta_multiclass(y_train, y_train_pred)
    print(f"The f beta score on train: {train_score}")

    # pt.draw_loss_curve(history)
    # pt.draw_accuracy_curve(history)
    # 保存模型
    # model.save('lstm_price_prediction_model.h5')
    # print("模型已保存为 'lstm_price_prediction_model.h5'")


if __name__ == "__main__":
    main()
