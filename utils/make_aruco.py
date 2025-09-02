#!/usr/bin/env python3
import cv2
import argparse

def main():
    p = argparse.ArgumentParser(
        description="Generate a single ArUco marker for printing")
    p.add_argument('--id',     type=int,   default=23, help="Marker ID")
    p.add_argument('--size',   type=int,   default=700,
                   help="Image size in pixels (square)")
    p.add_argument('--dict',   type=str,
                   default='DICT_4X4_50',
                   choices=[n for n in dir(cv2.aruco) if n.startswith('DICT_')],
                   help="ArUco dictionary")
    p.add_argument('--out',    type=str,   default='marker.png',
                   help="Output filename")
    args = p.parse_args()

    # pick the dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dict)
    )
    # generate the marker image
    marker = cv2.aruco.generateImageMarker(
        aruco_dict, args.id, args.size
    )
    cv2.imwrite(args.out, marker)
    print(f"Saved marker ID={args.id} to {args.out}")

if __name__=='__main__':
    main()
