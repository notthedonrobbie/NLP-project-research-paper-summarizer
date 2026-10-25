import * as echarts from "echarts";
import sharp from "sharp";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const result = JSON.parse(readFileSync(new URL("../results/full_text/evaluation.json", import.meta.url), "utf8"));
const profile = JSON.parse(readFileSync(new URL("../results/full_text/corpus_profile.json", import.meta.url), "utf8"));

const output = async (name, option, height = 650) => {
  const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width: 1200, height });
  chart.setOption({ animation: false, backgroundColor: "#ffffff", textStyle: { fontFamily: "Arial, sans-serif" }, ...option });
  const svg = chart.renderToSVGString();
  writeFileSync(fileURLToPath(new URL(`../report/figures/${name}.svg`, import.meta.url)), svg);
  await sharp(Buffer.from(svg)).png().toFile(fileURLToPath(new URL(`../report/figures/${name}.png`, import.meta.url)));
  chart.dispose();
};

const cutoffs = ["k1", "k3", "k5", "k10"];
await output("full-text-retrieval", {
  title: { text: "Known-item title-to-full-text retrieval", subtext: `${result.papers} title queries; title fields excluded from indexed papers`, left: "center" },
  color: ["#526d9b", "#376f65"], legend: { bottom: 10 },
  grid: { left: 80, right: 40, top: 110, bottom: 90 },
  xAxis: { type: "category", data: ["1", "3", "5", "10"], name: "Cut-off k", nameLocation: "middle", nameGap: 38 },
  yAxis: { type: "value", name: "Hit rate (0–1)", min: 0.85, max: 1.0, nameLocation: "middle", nameGap: 50 },
  series: ["bm25", "tfidf"].map((method) => ({
    name: method.toUpperCase(), type: "line", symbolSize: 9, lineStyle: { width: 3 },
    data: cutoffs.map((k) => Number(result.retrieval.methods[method].hit_rate[k].mean.toFixed(3))),
    label: { show: true, position: "top", formatter: "{c}" },
  })),
});

await output("full-text-summarization", {
  title: { text: "Six-sentence extractive summarization", subtext: "Macro-average ROUGE against author abstracts; abstract and references excluded from input", left: "center" },
  color: ["#c68a3c", "#7b658e"], legend: { bottom: 10 },
  grid: { left: 80, right: 40, top: 110, bottom: 90 },
  xAxis: { type: "category", data: ["ROUGE-1", "ROUGE-2", "ROUGE-L"] },
  yAxis: { type: "value", name: "F1 score (0–1)", min: 0, max: 0.4, nameLocation: "middle", nameGap: 50 },
  series: ["lead", "textrank"].map((method) => ({
    name: method === "lead" ? "LEAD" : "TextRank", type: "bar", barMaxWidth: 90,
    data: ["rouge1", "rouge2", "rougeL"].map((metric) => Number(result.summarization.methods[method].macro_average[metric].toFixed(3))),
    label: { show: true, position: "top", formatter: "{c}" },
  })),
});

const nodes = [
  ["600 CSV records", 65, 290, "#526d9b"], ["PDF acquisition", 240, 290, "#526d9b"],
  ["595 full papers\n12,836 pages", 420, 290, "#7b658e"], ["BM25 / TF-IDF", 630, 155, "#376f65"],
  ["Ranked papers", 850, 155, "#376f65"], ["Abstract / references\nexcluded", 630, 425, "#c68a3c"],
  ["LEAD / TextRank", 850, 425, "#c68a3c"], ["Metrics + source links", 1060, 290, "#344454"],
].map(([name, x, y, color]) => ({ name, x, y, symbolSize: [160, 62], itemStyle: { color }, label: { color: "#fff", fontSize: 13, lineHeight: 17 } }));
const links = [
  ["600 CSV records", "PDF acquisition"], ["PDF acquisition", "595 full papers\n12,836 pages"],
  ["595 full papers\n12,836 pages", "BM25 / TF-IDF"], ["BM25 / TF-IDF", "Ranked papers"],
  ["Ranked papers", "Metrics + source links"], ["595 full papers\n12,836 pages", "Abstract / references\nexcluded"],
  ["Abstract / references\nexcluded", "LEAD / TextRank"], ["LEAD / TextRank", "Metrics + source links"],
].map(([source, target]) => ({ source, target }));
await output("full-text-architecture", {
  title: { text: "PaperTrail full-text evaluation architecture", subtext: `${profile.extracted_words.total.toLocaleString("en-US")} extracted words; five failed records excluded with documented errors`, left: "center" },
  xAxis: { min: -20, max: 1160, show: false }, yAxis: { min: 75, max: 505, show: false },
  series: [{ type: "graph", layout: "none", roam: false, edgeSymbol: ["none", "arrow"], edgeSymbolSize: 10, label: { show: true }, lineStyle: { color: "#a9b2bb", width: 2 }, data: nodes, links }],
}, 580);

console.log("Rendered three full-text figures in report/figures/");
