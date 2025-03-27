import numpy as np

class DescriptorInterface:
    def __init__(self, values):
        self._values = np.array(values, dtype=np.float32)

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_vals):
        arr = np.array(new_vals, dtype=np.float32)
        if len(arr.shape) != 1:
            raise ValueError("Descriptor values must be a 1D array.")
        self._values = arr

    def distance(self, other):
        raise NotImplementedError("Must implement in subclass.")


class GeometricFeature(DescriptorInterface):
    descriptor_type = "geometric"

    def distance(self, other):
        if not isinstance(other, GeometricFeature):
            raise TypeError("Distance comparison must be with another GeometricFeature.")
        return np.array([
            (2 * abs(a - b)) / (1 + abs(a) + abs(b))
            for a, b in zip(self.values, other.values)
        ])


class ZernikeMoment(DescriptorInterface):
    descriptor_type = "zernike"

    def distance(self, other):
        if not isinstance(other, ZernikeMoment):
            raise TypeError("Distance comparison must be with another ZernikeMoment.")
        return np.abs(self.values - other.values)


class BioZernikeMoment:
    descriptor_type = "biozernike"

    def __init__(self, geometric_values, zernike_values):
        self.geom = GeometricFeature(geometric_values)
        self.zern = ZernikeMoment(zernike_values)

    def distance_vector(self, other):
        return np.concatenate([
            self.geom.distance(other.geom),
            self.zern.distance(other.zern)
        ])

    def bio_vector_length(self):
        return len(self.geom.distance(self.geom)) + len(self.zern.distance(self.zern))
