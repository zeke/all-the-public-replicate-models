# all-the-public-replicate-models

Metadata for all the public models on Replicate, bundled up into an npm package.

## Installation

```sh
npm install all-the-public-replicate-models
```

## Usage

Full-bodied usage (all the metadata, ~17MB)

```js
import models from 'all-the-public-replicate-models'

console.log(models)
```

Lite usage (just the basic metadata, ~375K):

```js
import models from 'all-the-public-replicate-models/lite'

console.log(models)
```

Find the top 10 models by run count:

```js
import models from 'all-the-public-replicate-models'
import {chain} from 'lodash-es'

const mostRun = chain(models).orderBy('run_count', 'desc').take(10).value()
console.log({mostRun})
```
