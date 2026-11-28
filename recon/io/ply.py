# SPDX-License-Identifier: Apache-2.0
"""NumPy scalar-vertex PLY reader/writer; ASCII and binary, no mesh/list properties."""

from pathlib import Path
import numpy as np

TYPES = {
    'char': 'i1',
    'uchar': 'u1',
    'short': 'i2',
    'ushort': 'u2',
    'int': 'i4',
    'uint': 'u4',
    'float': 'f4',
    'double': 'f8',
    'int8': 'i1',
    'uint8': 'u1',
    'int16': 'i2',
    'uint16': 'u2',
    'int32': 'i4',
    'uint32': 'u4',
    'float32': 'f4',
    'float64': 'f8',
}


def write_ply(path, vertices, binary=True):
    """Write a structured array preserving scalar properties (including Gaussian SH)."""
    vertices = np.asarray(vertices)
    if vertices.ndim != 1 or vertices.dtype.names is None:
        raise ValueError('Expected one-dimensional structured vertex array')
    names = vertices.dtype.names
    rev = {
        'i1': 'char',
        'u1': 'uchar',
        'i2': 'short',
        'u2': 'ushort',
        'i4': 'int',
        'u4': 'uint',
        'f4': 'float',
        'f8': 'double',
    }
    properties = []
    for name in names:
        dt = vertices.dtype[name]
        key = dt.kind + str(dt.itemsize)
        if key not in rev or dt.subdtype or not name.isidentifier():
            raise ValueError(f'Unsupported PLY property: {name} {dt}')
        properties.append(f'property {rev[key]} {name}')
    fmt = 'binary_little_endian' if binary else 'ascii'
    header = '\n'.join(['ply', f'format {fmt} 1.0', f'element vertex {len(vertices)}', *properties, 'end_header', ''])
    with Path(path).open('wb') as f:
        f.write(header.encode('ascii'))
        if binary:
            vertices.astype(np.dtype([(n, vertices.dtype[n].newbyteorder('<')) for n in names])).tofile(f)
        else:
            for row in vertices:
                f.write((' '.join(str(row[n]) for n in names) + '\n').encode('ascii'))


def read_ply(path):
    with Path(path).open('rb') as f:
        if f.readline() != b'ply\n':
            raise ValueError('Not a PLY file')
        fmt, count, props, element = None, None, [], None
        for _ in range(10000):
            line = f.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            words = line.decode('ascii').strip().split()
            if not words or words[0] in ('comment', 'obj_info'):
                continue
            if words[0] == 'format':
                fmt = words[1]
            elif words[0] == 'element':
                element = words[1]
                if element != 'vertex':
                    raise ValueError('Only vertex PLY supported (no mesh elements)')
                count = int(words[2])
            elif words[0] == 'property':
                if element != 'vertex' or len(words) != 3 or words[1] not in TYPES:
                    raise ValueError('Only scalar vertex properties supported')
                props.append((words[2], TYPES[words[1]]))
            elif words[0] == 'end_header':
                break
        else:
            raise ValueError('PLY header too long')
        if count is None or count < 0 or not props or fmt not in ('ascii', 'binary_little_endian', 'binary_big_endian'):
            raise ValueError('Invalid PLY header')
        endian = '>' if fmt == 'binary_big_endian' else '<'
        dtype = np.dtype([(n, endian + t) for n, t in props])
        if fmt == 'ascii':
            rows = []
            for _ in range(count):
                fields = f.readline().split()
                if len(fields) != len(props):
                    raise ValueError('Truncated or malformed PLY row')
                rows.append(tuple(fields))
            return np.array(rows, dtype=dtype)
        data = np.fromfile(f, dtype=dtype, count=count)
        if len(data) != count:
            raise ValueError('Truncated PLY body')
        return data


def point_vertices(points, colors):
    result = np.empty(
        len(points), dtype=[(n, '<f4') for n in ('x', 'y', 'z')] + [(n, 'u1') for n in ('red', 'green', 'blue')]
    )
    for j, n in enumerate(('x', 'y', 'z')):
        result[n] = points[:, j]
    for j, n in enumerate(('red', 'green', 'blue')):
        result[n] = np.rint(np.clip(colors[:, j], 0, 1) * 255).astype('u1')
    return result
