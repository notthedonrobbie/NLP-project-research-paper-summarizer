import * as echarts from "echarts";
import { readFileSync, writeFileSync } from "node:fs";

const result = JSON.parse(readFileSync(new URL("../results/pilot/aggregate.json", import.meta.url), "utf8"));
const output = (name, option, height = 650) => {
  const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width: 1200, height });
  chart.setOption({ animation: false, backgroundColor: "#ffffff", textStyle: { fontFamily: "Arial, sans-serif" }, ...option });
  writeFileSync(new URL(`../report/figures/${name}.svg`, import.meta.url), chart.renderToSVGString());
  chart.dispose();
};

const colors = { bm25: "#526d9b", tfidf: "#376f65", lead: "#c68a3c", textrank: "#7b658e" };
const metrics = ["rouge1", "rouge2", "rougeL"];
output("summarization-comparison", {
  title: { text: "Extractive summarization: macro-average ROUGE F1", subtext: "Three papers; independently written reference summaries", left: "center" },
  color: [colors.lead, colors.textrank],
  legend: { bottom: 10 },
  grid: { left: 75, right: 45, top: 105, bottom: 85 },
  xAxis: { type: "category", data: ["ROUGE-1", "ROUGE-2", "ROUGE-L"], axisLabel: { fontSize: 14 } },
  yAxis: { type: "value", name: "F1 score (0–1)", min: 0, max: 0.4, nameLocation: "middle", nameGap: 48 },
  series: ["lead", "textrank"].map((method) => ({
    name: method === "lead" ? "LEAD" : "TextRank",
    type: "bar",
    barGap: "12%",
    data: metrics.map((metric) => Number(result.summarization[method].macro_average[metric].toFixed(3))),
    label: { show: true, position: "top", formatter: "{c}" },
  })),
});

const ks = ["k1", "k3", "k5"];
output("retrieval-comparison", {
  title: { text: "Retrieval precision and recall across cut-offs", subtext: "15 answerable questions; fixed 1000-character chunks", left: "center" },
  color: [colors.bm25, colors.tfidf, "#8ca2c7", "#73a89e"],
  legend: { bottom: 8 },
  grid: { left: 75, right: 45, top: 105, bottom: 95 },
  xAxis: { type: "category", data: ["k = 1", "k = 3", "k = 5"], name: "Retrieved passages", nameLocation: "middle", nameGap: 35 },
  yAxis: { type: "value", name: "Mean score (0–1)", min: 0, max: 0.8, nameLocation: "middle", nameGap: 48 },
  series: ["bm25", "tfidf"].flatMap((method) => [
    { name: `${method.toUpperCase()} Precision`, type: "line", symbolSize: 9, lineStyle: { width: 3 }, data: ks.map((k) => Number(result.retrieval[method].question_level[k].mean_precision.toFixed(3))) },
    { name: `${method.toUpperCase()} Recall`, type: "line", symbol: "diamond", symbolSize: 10, lineStyle: { width: 3, type: "dashed" }, data: ks.map((k) => Number(result.retrieval[method].question_level[k].mean_recall.toFixed(3))) },
  ]),
});

const nodes = [
  ["PDF upload", 65, 290, "#526d9b"], ["Page extraction", 220, 290, "#526d9b"],
  ["Cleaning + sections", 390, 290, "#526d9b"], ["Page-aware chunks", 565, 290, "#7b658e"],
  ["Retriever", 735, 190, "#376f65"], ["Grounded answer", 905, 190, "#376f65"],
  ["Structured summary", 735, 420, "#c68a3c"], ["Streamlit UI + sources", 1060, 290, "#344454"],
].map(([name, x, y, color]) => ({ name, x, y, symbolSize: [140, 58], itemStyle: { color }, label: { color: "#fff", fontSize: 13, lineHeight: 17 } }));
const links = [
  ["PDF upload", "Page extraction"], ["Page extraction", "Cleaning + sections"],
  ["Cleaning + sections", "Page-aware chunks"], ["Page-aware chunks", "Retriever"],
  ["Retriever", "Grounded answer"], ["Grounded answer", "Streamlit UI + sources"],
  ["Page-aware chunks", "Structured summary"], ["Structured summary", "Streamlit UI + sources"],
].map(([source, target]) => ({ source, target }));
output("system-architecture", {
  title: { text: "PaperTrail processing architecture", subtext: "Every retrieval and summary path remains traceable to PDF pages", left: "center" },
  xAxis: { min: -10, max: 1140, show: false },
  yAxis: { min: 90, max: 500, show: false },
  series: [{ type: "graph", layout: "none", roam: false, edgeSymbol: ["none", "arrow"], edgeSymbolSize: 10, label: { show: true }, lineStyle: { color: "#a9b2bb", width: 2 }, data: nodes, links }],
}, 580);

console.log("Rendered three report figures in report/figures/");
