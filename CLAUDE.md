# CLAUDE.md



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
必ずserena MCPが日本語で対応すること！
日本語で応答すること！
最小機能ごとに実装し→動作確認を繰り返すことで少しずつ着実に実装すること！
仕様書の前提と実際の実装の観察結果が異なる場合は仕様書作成元のAIに指摘・質問する必要があるので、作業を止めて、Query prompt作成すること！
Web searchの際は、必ずo3-query MCPを利用すること！

## o3-query MCP について
エラーや未知の質問、技術的な調査が必要な場合は**必ず**o3-queryツールを呼び出し、
返ってきたクエリを私に表示して停止してください。
必ずクエリ生成→人間が結果を入力のフローに従ってください
※Queryの最後に「これらについて、公式実装を参考にした解決策を教えてください。」という文言を入れること

### 使用例
- 技術的なエラーの解決方法を調べたい時
- 最新の実装方法を調査したい時  
- 公式ドキュメントやベストプラクティスを確認したい時
- 具体的なライブラリやモデルの使用方法を調べたい時

[Introduction]
1．test_v2_implementation.pyやtest_with_masks_only.pyなどで、QwenVL→SAM2.1→Unidepth v2の流れで料理ごとの体積予測している。
2．Unidepth v2を最新のAppleのDepth Proに入れ替え、Depth Proに最適化されたアルゴリズムで体積予測をtest_depthpro_implementation.pyやtest_depthpro_optimized.pyで予測した。

1, 2両者において、体積予測結果が不十分だったため、Unidepth v2とDepth Anthing V2をNutrition 5KでFinetuningし、体積予測を比較したい。
→と思ったが、まずはDepth Anthing V2で同様に体積予測をTestしたい。

[命令]

md_files/DA_v2_spec.mdを参考に実装をすすめて。Testする画像はnutrition5kのサンプルを一つ選んでTestして。その際、DA_v2_spec.mdの情報で足りない場合は、その都度Query prompt作成して保存して（その後こちらで回答を共有します）。

README.mdの同様の環境とPythonを用いるぜんていですすめること。



機能ごとに少しずつ実装を行い、適宜テストを行い実際に動くことを確認して次の機能を実装するように進めること。

もしわからない部分があれば、積極的にAIにクエリするのでプロンプトを作成して。