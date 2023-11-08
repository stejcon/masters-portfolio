{
  description = "Application packaged using poetry2nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    # NOTE: If torch with cuda takes too long to build (took 6 hours with an i7 12th gen laptop) change cuda support to false
    pkgs = import nixpkgs { inherit system; config.allowUnfree = true; config.cudaSupport=true; };
    pyEnv = pkgs.python3.withPackages(ps: with ps; [ torch torchvision onnx numpy pillow matplotlib jupyter ipython ]);
  in {
    formatter.${system} = pkgs.alejandra;
    devShells.${system} = {
      default = pkgs.mkShell {
        packages = [ pyEnv ];
        shellHook = "jupyter notebook";
      };
    };
  };
}
