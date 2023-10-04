import Replicate from "replicate";
const replicate = new Replicate();
import {unset} from "lodash-es";

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

  process.stdout.write(JSON.stringify(models, null, 2))
}

main()
