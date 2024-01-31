import { readFile } from 'fs/promises';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
const __dirname = dirname(fileURLToPath(import.meta.url));
const models = JSON.parse(await readFile(join(__dirname, 'models.json'), 'utf8'));

export default models;