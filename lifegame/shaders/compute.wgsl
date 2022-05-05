[[block]]
struct Cells {
    data: [[stride(4)]] array<u32>;
};
[[group(0), binding(0)]]
var<storage, read> input: Cells;
[[group(0), binding(1)]]
var<storage, read_write> output: Cells;


fn cell_to_index(x: i32, y: i32) -> u32 {
    return u32((x % 250) + 250* (y % 250));
}
fn index_to_cell(ind: u32) -> vec2<u32> {
    let x = ind % 250u;
    let y = ind / 250u;
    return vec2<u32>(x, y);
}

[[stage(compute), workgroup_size(64, 1, 1)]]
fn main([[builtin(global_invocation_id)]] id: vec3<u32>) {
    let total = arrayLength(&input.data);
    let index: u32 = id.x;
    if (index >= total) {
        return;
    }
    let loc = index_to_cell(index);
    let x = i32(loc.x);
    let y = i32(loc.y);

    let sum = input.data[cell_to_index(x - 1, y - 1)]
            + input.data[cell_to_index(x - 1, y)]
            + input.data[cell_to_index(x - 1, y + 1)]
            + input.data[cell_to_index(x, y - 1)]
            + input.data[cell_to_index(x, y + 1)]
            + input.data[cell_to_index(x + 1, y - 1)]
            + input.data[cell_to_index(x + 1, y)]
            + input.data[cell_to_index(x + 1, y + 1)];
    var val = 0u;
    if (sum == 2u) {
        val = input.data[index];
    }
    if (sum == 3u) {
        val = 1u;
    }
    output.data[index] = val;
    return;
}
