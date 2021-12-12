use nannou::image::{self, GenericImageView};
use nannou::prelude::*;

use kd_tree::{self, KdPoint, KdTree};
use typenum::U2;

use ndarray::{self, Array2};

fn main() {
    nannou::app(model).update(update).run();
}

const N_PARTICLE: u16 = 8192;
const N_NEIGHBOR: usize = 10;
const PIC_R: u32 = 5;
const Q_CHARGE: f32 = 0.2;
const TIME_DELTA: f32 = 0.001;
const EPSILON: f32 = 0.00001;
const BLANK_LEVEL: f32 = 0.99;
const DOT_SIZE: f32 = 1.5;

#[derive(Clone)]
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}

impl Particle {
    fn new(x: f32, y: f32) -> Self {
        Particle {
            x, y,
            vx: 0.0,
            vy: 0.0,
        }
    }
}

impl KdPoint for Particle {
    type Scalar = f32;
    type Dim = U2;
    fn at(&self, k: usize) -> f32 {
        match k {
            0 => self.x,
            1 => self.y,
            _ => unreachable!()
        }
    }
}

struct Model {
    points: Vec<Particle>,
    n: u16,
    image: Array2<f32>,
    tree: KdTree<[f32; 2]>
}

impl Model {
    fn init(n: u16, img: Array2<f32>) -> Self {
        let mut points = vec![];
        for _ in 0..n {
            let x = random_range(-1.0, 1.0);
            let y = random_range(-1.0, 1.0);
            points.push(Particle::new(x, y));
        }
        let tree = KdTree::build_by_ordered_float(points.iter().map(|p| [p.x, p.y]).collect());
        Model { points, n, image: img, tree}
    }
}

fn length(x: f32, y: f32) -> f32 {
    (x.powf(2.0) + y.powf(2.0)).powf(0.5)
}

#[allow(unused)]
fn model(app: &App) -> Model {
    let assets = app.assets_path().unwrap();

    //let img_name = "Mona_Lisa.jpg";
    //let img_name = "cat_square.jpg";
    //let img_name = "namiura.jpg";
    let img_name = "black_cat.jpg";
    let img = image::open(assets.join("images").join(img_name)).unwrap();
    let (w, h) = img.dimensions();
    app.new_window()
        .size(w, h)
        .title("cat")
        .view(view)
        .key_pressed(key_pressed)
        .build()
        .unwrap();
    let mut img_mat: Array2<f32> = Array2::zeros((w as usize, h as usize));
    let ib = img.to_rgb8();
    let mut bmp_charge_total: f32 = 0.0;
    for (x, y, p) in ib.enumerate_pixels() {
        let charge = BLANK_LEVEL - (0.2989 * p[0] as f32 + 0.5870 * p[1] as f32 + 0.1140 * p[2] as f32) / 255.0;
        bmp_charge_total += charge;
        img_mat[[x as usize, y as usize]] = charge;
    }
    let area_r = (PIC_R + 1).pow(2) as f32 / (w * h) as f32;
    let mut bmp_charge = bmp_charge_total * area_r;
    let neighbor_dot_charge = N_NEIGHBOR as f32 * Q_CHARGE;
    bmp_charge = neighbor_dot_charge / bmp_charge;

    let img_mul = img_mat * bmp_charge;
    Model::init(N_PARTICLE, img_mul)
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(WHITE);
    let wr = app.window_rect();
    let l = wr.left();
    let r = wr.right();
    let b = wr.bottom();
    let t = wr.top();
    for p in model.points.iter() {
        draw.ellipse()
            .x_y(map_range(p.x, -1.0, 1.0, l, r), map_range(p.y, -1.0, 1.0, b, t))
            .radius(DOT_SIZE)
            .rgba8(0, 0, 0, 255);
    }
    draw.to_frame(app, &frame).unwrap();
}

fn rotate(v: f32) -> f32 {
    return if v > 1.0 {
        v - 2.0
    } else if v < -1.0 {
        v + 2.0
    } else {
        v
    }
}

fn update(app: &App, model: &mut Model, _: Update) {
    if app.elapsed_frames() == 250 {
        println!("updated 250 times");
    }
    if app.elapsed_frames() >= 250 {
        return
    }
    let dt = TIME_DELTA;
    let eps = EPSILON;
    let d_max = 0.005;
    let v_max = d_max / dt;
    let radius = PIC_R;
    let w = model.image.nrows() as u32;
    let h = model.image.ncols() as u32;
    for i in 0..model.n {
        let mut fx = 0.0;
        let mut fy = 0.0;
        let mut pi = model.points.get_mut(i as usize).unwrap();

        let pix = (0.5 * (1.0 + pi.x) * w as f32).ceil() as u32;
        let piy = h.saturating_sub((0.5 * (1.0 + pi.y) * h as f32).floor() as u32);

        for xi in pix.saturating_sub(radius)..(pix + radius).min(w) {
            for yi in piy.saturating_sub(radius)..(piy + radius).min(h) {
                let dx = rotate(pi.x - (2.0 * xi as f32 / w as f32 - 1.0));
                let dy = rotate(pi.y - (1.0 - 2.0 * yi as f32 / h as f32));
                let d2 = dx.powf(2.0) + dy.powf(2.0) + 0.00003;
                let d = d2.powf(0.5);
                let q = Q_CHARGE / d2;
                let pq = model.image[[xi as usize, yi as usize]];
                if d.abs() > eps {
                    fx -= pq * q * dx / d;
                    fy -= pq * q * dy / d;
                }
            }
        }
        for pj in model.tree.nearests(&[pi.x, pi.y], N_NEIGHBOR).iter() {
            let dx = rotate(pi.x - pj.item[0]);
            let dy = rotate(pi.y - pj.item[1]);
            let d2 = dx.powf(2.0) + dy.powf(2.0) + 0.00003;
            let d = d2.powf(0.5);
            let q = Q_CHARGE / d2;
            if d.abs() > eps {
                fx += Q_CHARGE * q * dx / d;
                fy += Q_CHARGE * q * dy / d;
            }
        }
        let mut dvx = dt * fx;
        let mut dvy = dt * fy;
        let vl = length(dvx, dvy);
        if vl > v_max {
            dvx *= v_max / vl;
            dvy *= v_max / vl;
        }
        pi.vx = 0.95 * pi.vx + dvx;
        pi.vy = 0.95 * pi.vy + dvy;
        let mut dpx = dt * pi.vx + 0.5 * dt.pow(2) as f32 * fx;
        let mut dpy = dt * pi.vy + 0.5 * dt.pow(2) as f32 * fy;
        let dl = length(dpx, dpy);
        if dl > d_max {
            dpx *= d_max / dl;
            dpy *= d_max / dl;
        }
        pi.x = rotate(pi.x + dpx);
        pi.y = rotate(pi.y + dpy);
    }
    let tree = KdTree::build_by_ordered_float(model.points.iter().map(|p| [p.x, p.y]).collect());
    model.tree = tree;
}

fn key_pressed(app: &App, _: &mut Model, key: Key) {
    match key {
        Key::S => {
            app.main_window().capture_frame(app.exe_name().unwrap() + ".png");
            println!("saved!");
        },
        Key::Q => app.quit(),
        _ => {}
    }
}
