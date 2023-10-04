import models from './index.mjs'
import {chain} from 'lodash-es'

const mostRunModels = chain(models)
  .orderBy('run_count', 'desc')
  .take(10)
  .value()

for (const model of mostRunModels) {
  console.log(model.url)
}