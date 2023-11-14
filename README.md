# all-the-public-replicate-models

Metadata for all the public models on Replicate, bundled up into an npm package.

## Installation

```sh
npm install all-the-public-replicate-models
```

## Usage (as a library)

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

## Usage (as a CLI)

The CLI dumps the model metadata to standard output as a big JSON object:

```command
$ npx all-the-public-replicate-models
```

The output will be:

```
[
  {...},
  {...},
  {...},
]
```

You can use [jq](https://stedolan.github.io/jq/) to filter the output. Here's an example that finds all the whisper models and sorts them by run count:

```command
npx all-the-public-replicate-models | jq -r 'map(select(.name | contains("whisper"))) | sort_by(.run_count) | reverse | .[] | "\(.url) \(.run_count)"'
```

- https://replicate.com/openai/whisper 3790120
- https://replicate.com/m1guelpf/whisper-subtitles 38020
- https://replicate.com/hnesk/whisper-wordtimestamps 28889
- https://replicate.com/alqasemy2020/whisper-jax 20296
- https://replicate.com/wglodell/cog-whisperx-withprompt 19326
- https://replicate.com/daanelson/whisperx 15528
...


Or you can dump all the model data to a file:

```command
npx all-the-public-replicate-models > models.json
```