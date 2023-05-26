import argparse
import torch
from radiance_fields.ngp import NGPradianceField


if __name__ == "__main__":

    device = "cuda:0"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=[
            # nerf synthetic
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
            # nsvf synthetic
            "Wineholder",
            "Steamtrain",
            "Spaceship",
            "Palace",
            "Bike",
            "Robot",
            "Lifestyle",
            "Toad",
            # BlendedMVS
            "Jade",
            "Fountain",
            "Statues",
            "Character",
            # TanksAndTemples
            "Barn",
            "Caterpillar",
            "Family",
            "Ignatius",
            "Truck",
            # mipnerf360 unbounded
            "garden",
            "bicycle",
            "bonsai",
            "counter",
            "kitchen",
            "room",
            "stump",
            # llff
            "fern",
            "flower",
            "fortress",
            "horns",
            "leaves",
            "orchids",
            "room",
            "trex",
        ],
        help="which scene to use",
    )

    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
        help="delimited list input",
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="whether to use unbounded rendering",
    )
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--n_level", type=int, default=16)
    parser.add_argument("--hash_size", type=int, default=19)
    parser.add_argument("--base_layer", type=int, default=1)
    parser.add_argument("--base_dim", type=int, default=64)
    parser.add_argument("--head_layer", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--geo_feat_dim", type=int, default=15)

    parser.add_argument("--nsvf", action='store_true')
    args = parser.parse_args()

    render_n_samples = 1024

    if args.nsvf:
        aabb_path = '/'.join(['/home/eic/yzf/TanksAndTemple', args.scene, 'bbox.txt'])
        with open(aabb_path, 'r') as f:
            line = f.readline()
            args.aabb = [float(w) for w in line.strip().split()][:-1]
        print("Override args.aabb by NSVF/BlendedMVS/TanksAndTemples provided BBOX: " + str(args.aabb))


    radiance_field = NGPradianceField(
        aabb=args.aabb,
        unbounded=args.unbounded,
        n_levels=args.n_level,
        log2_hashmap_size=args.hash_size,
        base_layer=args.base_layer,
        base_dim=args.base_dim,
        head_layer=args.head_layer,
        head_dim=args.head_dim,
        geo_feat_dim=args.geo_feat_dim,
    ).to(device)

    radiance_field.load_state_dict(torch.load('/'.join([args.load_path, 'state_dict.pt'])))
    torch.save(radiance_field, '/'.join([args.save_path, 'model.pt']))