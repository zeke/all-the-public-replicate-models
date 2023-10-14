import { promises as fs } from 'fs';
import oldModels from '../models.old.json' assert { type: "json" };
import newModels from '../models.json' assert { type: "json" };

async function compareAndGenerateMD() {
    const newModelUrls = newModels.map(model => model.url);
    const oldModelUrls = oldModels.map(model => model.url);

    const addedModels = newModels.filter(model => !oldModelUrls.includes(model.url));
    const removedModels = oldModels.filter(model => !newModelUrls.includes(model.url));

    const activeModels = newModels.map(newModel => {
        const oldModel = oldModels.find(m => m.url === newModel.url);
        const runCountDiff = oldModel ? newModel.run_count - oldModel.run_count : newModel.run_count;
        return { ...newModel, runCountDiff };
    })
    .filter(model => model.runCountDiff > 100)
    .sort((a, b) => b.runCountDiff - a.runCountDiff);

    const risingStars = newModels.map(model => {
        const oldModel = oldModels.find(m => m.url === model.url);
        const runCountDiff = oldModel ? model.run_count - oldModel.run_count : model.run_count;
        const percentageIncrease = runCountDiff / model.run_count * 100;
        return { ...model, runCountDiff, percentageIncrease };
    })
    .sort((a, b) => b.percentageIncrease - a.percentageIncrease)
    .slice(0, 50);

    let markdownContent = '# Model Stats\n';

    // New Models
    markdownContent += '## New Models\n';
    if (addedModels.length > 0) {
        for (let model of addedModels) {
            markdownContent += `- ${model.url}\n`;
        }
    } else {
        markdownContent += 'No new models today.\n';
    }

    // Removed Models
    markdownContent += '\n## Removed Models\n';
    if (removedModels.length > 0) {
        for (let model of removedModels) {
            markdownContent += `- ${model.url}\n`;
        }
    } else {
        markdownContent += 'No models were removed today.\n';
    }

    // Rising Stars
    markdownContent += '\n## Rising Stars\n';
    if (risingStars.length > 0) {
        markdownContent += '| Model | Description | Runs Today | Runs Total | % of Total |\n|-------|-------------|------------|------------|------------|\n';
        for (let model of risingStars) {
            const linkText = `${model.owner}/${model.name}`;
            const percentageDisplay = `${model.percentageIncrease.toFixed(2)}%`;
            markdownContent += `| [${linkText}](${model.url}) | ${model.description} | ${model.runCountDiff} | ${model.run_count} | ${percentageDisplay} |\n`;
        }
    } else {
        markdownContent += 'No rising stars today.\n';
    }

    // Active Models
    markdownContent += '\n## Active Models\n';
    if (activeModels.length > 0) {
        markdownContent += '| Model | Description | Runs in the last day |\n|-------|-------------|---------------------|\n';
        for (let model of activeModels) {
            const linkText = `${model.owner}/${model.name}`;
            markdownContent += `| [${linkText}](${model.url}) | ${model.description} | ${model.runCountDiff} |\n`;
        }
    } else {
        markdownContent += 'No active models today.\n';
    }

    // Write to stats.md
    await fs.writeFile('stats.md', markdownContent);
}

compareAndGenerateMD().catch(error => {
    console.error('Error generating stats:', error);
});
