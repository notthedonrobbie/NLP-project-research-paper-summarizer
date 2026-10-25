import * as echarts from "echarts";
import sharp from "sharp";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const read = (name) => JSON.parse(readFileSync(new URL(`../results/ai_research_papers/${name}`, import.meta.url), "utf8"));
const profile = read("dataset_profile.json");
const retrieval = read("known_item_retrieval.json");

const output = async (name, option, height = 650) => {
  const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width: 1200, height });
  chart.setOption({ animation: false, backgroundColor: "#ffffff", textStyle: { fontFamily: "Arial, sans-serif" }, ...option });
  const svg = chart.renderToSVGString();
  const svgPath = fileURLToPath(new URL(`../report/figures/${name}.svg`, import.meta.url));
  const pngPath = fileURLToPath(new URL(`../report/figures/${name}.png`, import.meta.url));
  writeFileSync(svgPath, svg);
  await sharp(Buffer.from(svg)).png().toFile(pngPath);
  chart.dispose();
};

const years = Object.keys(profile.papers_by_year);
await output("ai-dataset-year-distribution", {
  title: { text: "Paper records by publication year", subtext: "600 supplied arXiv-linked records; 2025 is partial through 10 April", left: "center" },
  color: ["#526d9b"],
  grid: { left: 75, right: 35, top: 105, bottom: 80 },
  xAxis: { type: "category", data: years, name: "Publication year", nameLocation: "middle", nameGap: 42, axisLabel: { fontSize: 13 } },
  yAxis: { type: "value", name: "Number of papers", nameLocation: "middle", nameGap: 48, minInterval: 1 },
  series: [{ name: "Papers", type: "bar", data: years.map((year) => profile.papers_by_year[year]), label: { show: true, position: "top", fontSize: 12 } }],
});

const cutoffs = ["k1", "k3", "k5", "k10"];
await output("ai-known-item-retrieval", {
  title: { text: "Known-item title-to-abstract retrieval", subtext: "600 title queries; one relevant abstract per query", left: "center" },
  color: ["#526d9b", "#376f65"],
  legend: { bottom: 10 },
  grid: { left: 75, right: 40, top: 105, bottom: 90 },
  xAxis: { type: "category", data: ["1", "3", "5", "10"], name: "Cut-off k", nameLocation: "middle", nameGap: 38 },
  yAxis: { type: "value", name: "Hit rate (0–1)", min: 0.8, max: 1.0, nameLocation: "middle", nameGap: 48 },
  series: ["bm25", "tfidf"].map((method) => ({
    name: method.toUpperCase(), type: "line", symbolSize: 9, lineStyle: { width: 3 },
    data: cutoffs.map((k) => Number(retrieval.methods[method].hit_rate[k].mean.toFixed(3))),
    label: { show: true, position: "top", formatter: "{c}" },
  })),
});

const nodes = [
  ["CSV: 600 records", 70, 290, "#526d9b"], ["Validate + clean", 245, 290, "#526d9b"],
  ["Abstract corpus", 420, 290, "#7b658e"], ["BM25 / TF-IDF", 610, 175, "#376f65"],
  ["Ranked papers + links", 850, 175, "#376f65"], ["LEAD / TextRank", 610, 410, "#c68a3c"],
  ["Abstract brief", 850, 410, "#c68a3c"], ["Inspection interface", 1060, 290, "#344454"],
].map(([name, x, y, color]) => ({ name, x, y, symbolSize: [155, 58], itemStyle: { color }, label: { color: "#fff", fontSize: 13, lineHeight: 17 } }));
const links = [
  ["CSV: 600 records", "Validate + clean"], ["Validate + clean", "Abstract corpus"],
  ["Abstract corpus", "BM25 / TF-IDF"], ["BM25 / TF-IDF", "Ranked papers + links"],
  ["Ranked papers + links", "Inspection interface"], ["Abstract corpus", "LEAD / TextRank"],
  ["LEAD / TextRank", "Abstract brief"], ["Abstract brief", "Inspection interface"],
].map(([source, target]) => ({ source, target }));
await output("ai-dataset-architecture", {
  title: { text: "Abstract-level research discovery architecture", subtext: "Source links preserve traceability; full-paper QA requires additional documents", left: "center" },
  xAxis: { min: -15, max: 1150, show: false }, yAxis: { min: 80, max: 500, show: false },
  series: [{ type: "graph", layout: "none", roam: false, edgeSymbol: ["none", "arrow"], edgeSymbolSize: 10, label: { show: true }, lineStyle: { color: "#a9b2bb", width: 2 }, data: nodes, links }],
}, 580);

console.log("Rendered three AI-paper dataset figures in report/figures/");
