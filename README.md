# シミュレーション管理

## 目的

本組織「Modeling-Coding-Automation-Project」にあるリポジトリで実行する時間的なシミュレーションを、効率的に行うために便利なスクリプトなどをまとめています。

## SimulationPlotterクラスの使い方

SimulationPlotter クラスは、時間系列データ（信号）の可視化を簡単に行うためのユーティリティです。基本的な流れは以下の通りです。

- インポートとインスタンス化
  - from visualize.simulation_plotter import SimulationPlotter
  - plotter = SimulationPlotter()

- 信号の追加
  - append / append_name / append_sequence / append_sequence_name を使い、時間ステップごとの値やシーケンスを登録します。
  - 例: plotter.append_sequence_name(time_array_and_signals, "input_signal")

- サブプロットへの割り当て
  - assign / assign_all を使って、表示したい信号とサブプロットの位置・スタイルを指定します。
  - 例: plotter.assign("input_signal", column=0, row=0, x_sequence=time_array, label="入力")

- プロットの表示
  - plotter.plot("図のタイトル") でウィンドウを表示します。pre_plot を使えば描画前の設定のみ行えます。
  - 表示中はマウススクロールでX軸方向に拡大縮小できます。
  - Ctrlを押しながらマウススクロールでY軸方向に拡大縮小できます。
  - 信号線をクリックすると、その点に対するカーソル情報を表示できます。
  - Dual cursor modeのチェックを入れると、2軸カーソルを表示できます。
    - 左クリック、右クリックで、それぞれのカーソルをそのマウスカーソル地点に出せます。

- ログ保存機能
  - plotterのクラスインスタンスを以下のように定義すると、
``` python
plotter = SimulationPlotter(activate_dump=True)
```
  - プロットデータをファイルに保存できます。
  - 保存したファイルをロードしてプロットするには、以下のように実行します。
``` python
plotter.plot(
    dump_file_path=r".\cache\simulation_plotter_dumps\SimulationPlotterData_20260110142854.npz",
)
```

## サポート

新規にissueを作成して、詳細をお知らせください。

## 貢献

コミュニティからのプルリクエストを歓迎します。もし大幅な変更を考えているのであれば、提案する修正についての議論を始めるために、issueを開くことから始めてください。

また、プルリクエストを提出する際には、関連するテストが必要に応じて更新または追加されていることを確認してください。

## ライセンス

[MIT License](./LICENSE.txt)
