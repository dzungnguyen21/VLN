from libs import *

NANOSECONDS_PER_SECOND = 1000000000

RIG_NAME = "head_front"
TRAJ_FILENAME = f"traj_{RIG_NAME}.txt"
CAMERA_BASE = "/camera/head_front_zed_onboard/left/color/rect"
IMAGE_TOPIC = CAMERA_BASE + "/image/compressed"
CAMERA_INFO_TOPICS = (CAMERA_BASE + "/image/camera_info", CAMERA_BASE + "/camera_info")


class CDRReader:
    HEADER_SIZE = 4

    def __init__(self, buffer):
        self.buffer = buffer
        self.endian = "<" if buffer[1] & 1 else ">"
        self.position = self.HEADER_SIZE

    def align_to(self, size_bytes):
        self.position += (-(self.position - self.HEADER_SIZE)) % size_bytes

    def read_scalar(self, struct_format, size_bytes):
        self.align_to(size_bytes)
        value = struct.unpack_from(self.endian + struct_format, self.buffer, self.position)[0]
        self.position += size_bytes
        return value

    def read_uint32(self):
        return self.read_scalar("I", 4)

    def read_int32(self):
        return self.read_scalar("i", 4)

    def read_string(self):
        length = self.read_uint32()
        text = bytes(self.buffer[self.position:self.position + max(0, length - 1)])
        self.position += length
        return text.decode("utf-8", "replace")

    def read_float64_array(self, count):
        self.align_to(8)
        values = np.frombuffer(self.buffer, self.endian + "f8", count, self.position)
        self.position += 8 * count
        return values.copy()

    def read_byte_sequence(self):
        length = self.read_uint32()
        data = bytes(self.buffer[self.position:self.position + length])
        self.position += length
        return data

    def read_header(self):
        seconds, nanoseconds = self.read_int32(), self.read_uint32()
        return seconds * NANOSECONDS_PER_SECOND + nanoseconds, self.read_string()


def header_stamp_ns(header_bytes):
    struct_format = "<iI" if header_bytes[1] & 1 else ">iI"
    seconds, nanoseconds = struct.unpack_from(struct_format, header_bytes, 4)
    return seconds * NANOSECONDS_PER_SECOND + nanoseconds


def decode_camera_info(buffer):
    reader = CDRReader(buffer)
    reader.read_header()
    height, width = reader.read_uint32(), reader.read_uint32()
    reader.read_string()
    distortion = reader.read_float64_array(reader.read_uint32())
    intrinsics = reader.read_float64_array(9).reshape(3, 3)
    return {
        "width": width,
        "height": height,
        "intrinsics": intrinsics,
        "rectified": bool(np.allclose(distortion, 0)),
    }


def decode_compressed_image(buffer):
    reader = CDRReader(buffer)
    reader.read_header()
    reader.read_string()
    return reader.read_byte_sequence()


class Bag:
    def __init__(self, database_path):
        self.path = Path(database_path)
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        topics = self.connection.execute("SELECT id, name FROM topics")
        self.topic_ids = {name: topic_id for topic_id, name in topics}

    @classmethod
    def open(cls, folder):
        database_files = sorted(Path(folder).glob("*.db3"))
        if not database_files:
            raise ValueError(f"no .db3 in {folder}")
        if len(database_files) > 1:
            raise ValueError(f"{folder} is a split recording "
                             f"({len(database_files)} .db3) — needs one file")
        return cls(database_files[0])

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception_info):
        self.close()

    def find_topic_id(self, *candidate_names):
        for name in candidate_names:
            if name in self.topic_ids:
                return self.topic_ids[name]
        raise ValueError(f"none of {candidate_names} in {self.path.parent.name}. "
                         f"Topics present: {', '.join(sorted(self.topic_ids))}")

    def camera_info(self):
        topic_id = self.find_topic_id(*CAMERA_INFO_TOPICS)
        row = self.connection.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1",
            (topic_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"camera_info topic is empty in {self.path.parent.name}")
        return decode_camera_info(bytes(row[0]))

    def image_index(self):
        topic_id = self.find_topic_id(IMAGE_TOPIC)
        rows = self.connection.execute(
            "SELECT substr(data,1,12), id FROM messages WHERE topic_id=? ORDER BY timestamp",
            (topic_id,),
        ).fetchall()
        if not rows:
            raise ValueError(f"image topic is empty in {self.path.parent.name}")

        stamped_rows = sorted((header_stamp_ns(bytes(header)), row_id) for header, row_id in rows)
        timestamps_ns = np.array([stamp for stamp, _ in stamped_rows], np.int64)
        row_ids = [row_id for _, row_id in stamped_rows]
        return timestamps_ns, row_ids

    def image(self, row_id):
        row = self.connection.execute("SELECT data FROM messages WHERE id=?", (row_id,)).fetchone()
        if row is None:
            raise ValueError(f"message {row_id} vanished from {self.path}")
        return Image.open(BytesIO(decode_compressed_image(bytes(row[0])))).convert("RGB")

    def iter_jpegs(self):
        topic_id = self.find_topic_id(IMAGE_TOPIC)
        for row in self.connection.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (topic_id,),
        ):
            buffer = bytes(row[0])
            yield header_stamp_ns(buffer[:12]), decode_compressed_image(buffer)
