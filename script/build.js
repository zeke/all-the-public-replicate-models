import Replicate from "replicate";
const replicate = new Replicate();
import {unset} from "lodash-es";
import fs from "fs";

async function main () {
  console.error("Fetching all public models from Replicate...")
  const models = []
  for await (const batch of replicate.paginate(replicate.models.list)) {
    process.stderr.write('.')
    models.push(...batch);
  }

  // remove some noisy fields that are not needed
  for (let i = 0; i < models.length; i++) {
    unset(models[i], 'default_example.logs')
    unset(models[i], 'default_example.urls')
    unset(models[i], 'default_example.webhook_completed')
    unset(models[i], 'latest_version.openapi_schema.paths')
  }
  
  const lite = models.map(model => {
    return {
      url: model.url,
      owner: model.owner,
      name: model.name,
      description: model.description,
      run_count: model.run_count,
      cover_image_url: model.cover_image_url
    }
  });

  fs.writeFileSync('models.json', JSON.stringify(models, null, 2))
  fs.writeFileSync('models-lite.json', JSON.stringify(lite, null, 2))
}

main()
