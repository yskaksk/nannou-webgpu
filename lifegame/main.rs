use nannou::prelude::*;
use nannou::rand::{
    self,
    distributions::{Distribution, Uniform},
    SeedableRng
};
use std::sync::{Arc, Mutex};

const CELLS_PER_ROW : usize = 250;
const CELLS_COUNT : usize = CELLS_PER_ROW * CELLS_PER_ROW;
const CELLS_PER_GROUP: u32 = 64;

struct Model {
    compute: Compute,
    cells: Arc<Mutex<Vec<u32>>>,
    fc: u32,
}

struct Compute {
    buffers: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
    pipeline: wgpu::ComputePipeline
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let w_id = app.new_window().size(1200, 1200).title("lifegame").view(view).build().unwrap();
    let window = app.window(w_id).unwrap();
    let device = window.device();

    let desc = wgpu::include_wgsl!("shaders/compute.wgsl");
    let shader = device.create_shader_module(&desc);

    let cells = create_initial_cells();
    let mut buffers = Vec::<wgpu::Buffer>::new();
    for i in 0..2 {
        buffers.push(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("cell buffer {}", i)),
            contents: bytemuck::cast_slice(&cells),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::MAP_READ,
        }));
    }
    let bind_group_layout = create_bind_group_layout(&device);
    let pipeline_layout = create_pipeline_layout(&device, &bind_group_layout);
    let pipeline = create_compute_pipeline(&device, &pipeline_layout, &shader);
    let mut bind_groups = Vec::<wgpu::BindGroup>::new();
    for i in 0..2 {
        bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers[i % 2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers[(i+1)%2].as_entire_binding(),
                },
            ],
            label: None,
        }));
    }
    let compute = Compute {
        buffers,
        bind_groups,
        pipeline
    };

    Model {
        compute,
        cells: Arc::new(Mutex::new(vec![0 as u32; CELLS_COUNT as usize])),
        fc: 0
    }
}

fn update(app: &App, model: &mut Model, _: Update) {
    let window = app.main_window();
    let device = window.device();
    let compute = &model.compute;
    let fc = model.fc as usize;

    let buffer_size = CELLS_COUNT as u64 * std::mem::size_of::<u32>() as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compute"),
    });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute pass"),
        });
        compute_pass.set_pipeline(&compute.pipeline);
        compute_pass.set_bind_group(0, &compute.bind_groups[fc % 2], &[]);
        compute_pass.dispatch(((CELLS_COUNT as f32) / (CELLS_PER_GROUP as f32)).ceil() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&compute.buffers[(fc + 1) % 2], 0, &output_buffer, 0, buffer_size);
    window.queue().submit(Some(encoder.finish()));

    let cells = Arc::clone(&model.cells);
    let future = async move {
        let slice = output_buffer.slice(..);
        if let Ok(_) = slice.map_async(wgpu::MapMode::Read).await {
            if let Ok(mut cells) = cells.lock() {
                let bytes = &slice.get_mapped_range()[..];
                let u32s = {
                    let len = bytes.len() / std::mem::size_of::<u32>();
                    let ptr = bytes.as_ptr() as *const u32;
                    unsafe { std::slice::from_raw_parts(ptr, len) }
                };
                cells.copy_from_slice(u32s);
            }
        }
        output_buffer.unmap();
    };
    async_std::task::spawn(future);

    model.fc += 1;
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.reset();
    draw.background().color(WHITE);
    let window = app.window(frame.window_id()).unwrap();
    let rect = window.rect();

    if let Ok(cells) = model.cells.lock() {
        for (i, cell) in cells.iter().enumerate() {
            if *cell == 1 {
                let (xi, yi) = index_to_loc(i);
                let x = map_range(xi, 0.0, 1.0, rect.left(), rect.right());
                let y = map_range(yi, 0.0, 1.0, rect.bottom(), rect.top());
                draw.rect()
                    .x_y(x, y)
                    .w_h(rect.w() / (CELLS_PER_ROW as f32), rect.h() / (CELLS_PER_ROW as f32))
                    .rgb(0.0, 0.0, 0.0);
            }
        }
    }
    draw.to_frame(app, &frame).unwrap();
}

fn index_to_loc(index: usize) -> (f32, f32) {
    let xi = index % CELLS_PER_ROW;
    let yi = index / CELLS_PER_ROW;
    let x = (1.0 / (2.0 * CELLS_PER_ROW as f32)) + (xi as f32) / (CELLS_PER_ROW as f32);
    let y = (1.0 / (2.0 * CELLS_PER_ROW as f32)) + (yi as f32) / (CELLS_PER_ROW as f32);
    return (x, y)
}

fn create_initial_cells() -> Vec<u32> {
    let mut cells : Vec<u32> = vec![0; CELLS_COUNT];
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let unif = Uniform::new_inclusive::<f32, f32>(0.0, 1.0);
    for i in 0..CELLS_COUNT {
        if unif.sample(&mut rng) < 0.2 {
            cells[i] = 1;
        }
    }
    return cells
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
   return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(CELLS_COUNT as _),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(CELLS_COUNT as _),
                },
                count: None,
            },
        ],
        label: None
    });
}

fn create_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("lifegame"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let desc = wgpu::ComputePipelineDescriptor {
        label: Some("lifegame"),
        layout: Some(layout),
        module: &shader,
        entry_point: "main",
    };
    device.create_compute_pipeline(&desc)
}
