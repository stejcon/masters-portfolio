{
  description = "Application packaged using poetry2nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
      config.cudaSupport = true;
    };
    matplotlib = pkgs.python3Packages.matplotlib.override {enableQt = true;};
    pyEnv = pkgs.python3.withPackages (ps: with ps; [torchvision-bin onnx numpy pillow jupyter matplotlib ipython scipy pyqt5]);
  in {
    formatter.${system} = pkgs.alejandra;
    devShells.${system} = {
      default = pkgs.mkShell {
        packages = [pyEnv pkgs.typst pkgs.jq];
        QT_PLUGIN_PATH = with pkgs.qt5; "${qtbase}/${qtbase.qtPluginPrefix}";
      };
    };
  };
}
