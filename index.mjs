
import { readFile } from 'fs/promises';

const models = JSON.parse(await readFile('./models.json', 'utf8'));


export default models;
