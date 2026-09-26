import { nodeResolve } from '@rollup/plugin-node-resolve';

export default {
    input: 'index.js',
    output: {
        file: 'index.cjs',
        format: 'cjs'
    },
    external: [
        'express',
        'colors',
        'ms'
    ],
    plugins: [nodeResolve()]
};