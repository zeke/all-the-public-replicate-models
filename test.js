import { expect } from 'chai';
import models from './index.mjs';

describe('models module', () => {
  it('should be an array', () => {
    expect(models).to.be.an('array');
  });

  it('should have at least 500 objects', () => {
    expect(models).to.have.lengthOf.at.least(500);
  });

  describe('first object in the array', () => {
    it('should have an owner property', () => {
      expect(models[0]).to.have.property('owner');
    });

    it('should have a name property', () => {
      expect(models[0]).to.have.property('name');
    });
  });
});
