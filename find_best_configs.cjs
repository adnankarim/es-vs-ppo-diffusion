const fs = require('fs');

const dims = [1, 2, 5, 10, 20, 30];
const paths = {
    es: 'public/es/all_results.json',
    ppo: 'public/ppo/all_results.json'
};

function findBest(data, dim, method) {
    if (!data[dim] || !data[dim][method]) return null;
    const updates = data[dim][method];
    let bestIdx = -1;
    let bestMAE = Infinity;
    let bestMetrics = {};

    updates.forEach((run, idx) => {
        if (run && typeof run.mae === 'number') {
            if (run.mae < bestMAE) {
                bestMAE = run.mae;
                bestIdx = idx;
                bestMetrics = run;
            }
        }
    });

    return {
        config: bestIdx,
        mae: bestMetrics.mae,
        mi: bestMetrics.mutual_information,
        kl: bestMetrics.kl_div_total
    };
}

try {
    const esData = JSON.parse(fs.readFileSync(paths.es, 'utf8'));
    const ppoData = JSON.parse(fs.readFileSync(paths.ppo, 'utf8'));

    const output = {};

    dims.forEach(dim => {
        output[dim] = {
            es: findBest(esData, dim, 'ES'),
            ppo: findBest(ppoData, dim, 'PPO')
        };
    });

    console.log(JSON.stringify(output, null, 2));

} catch (e) {
    console.error("Error reading/parsing files:", e);
}
