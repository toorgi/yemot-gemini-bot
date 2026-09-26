import request from 'supertest';
import qs from 'qs';
import crypto from 'crypto';

export class CallSimulator {
    #port;
    constructor (port) {
        this.#port = port;

        this.values = {
            ApiCallId: crypto.randomBytes(20).toString('hex'),
            ApiYFCallId: crypto.randomBytes(20).toString('hex'),
            ApiDID: '0772222770',
            ApiRealDID: '07722225555',
            ApiPhone: '0527000000',
            ApiExtension: '',
            ApiTime: new Date().getTime().toString()
        };
    }

    get (path = '/') {
        return request(`http://localhost:${this.#port}`).get(`${path}?${qs.stringify(this.values)}`);
    }

    post (path = '/') {
        return request(`http://localhost:${this.#port}`).post(`${path}?${qs.stringify(this.values)}`);
    }
}