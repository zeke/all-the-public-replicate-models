#!/usr/bin/env node

import models from './index.mjs';

process.stdout.write(JSON.stringify(models, null, 2));