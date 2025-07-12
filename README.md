## Adding new LDraw models to Unreal Engine

To add new models from the LDraw library to the Unreal Engine project:

1. Import the LDraw model into Blender using, for example, [TobyLobster/ImportLDraw](https://github.com/TobyLobster/ImportLDraw).
2. Adjust its pivot point so that it is centered within the object.
3. Export the model as an FBX file.
4. Import the FBX into Unreal Engine.

---

## Adding a new model to the generator

To add a new model to the generator:

1. Add a new entry to the `DT_LegoParts` data table.
2. Select the new 3D model.
3. Fill in the `UnlitColor` parameter.

---

## Increasing the number of LEGO bricks (classes)

To increase the number of distinguishable LEGO brick classes:

- Add a new segmentation mask based on a new `UnlitColor`.
- By using two segmentation masks, you can combine colors to achieve significantly more classes: instead of the current 46 classes (46 unlit colors), you'll have 46² = 2116 classes.

---

## Improving segmentation masks

To improve segmentation masks:

- Create a separate mask for each brick type, by hidding other classes.
- Calculate what percentage of each brick is visible in the final scene.
- This approach enables the generation of better masks and, consequently, more accurate bounding boxes.

---

## License Notice

This project includes files from **The LDraw Parts Library**, licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [ldraw.org](https://www.ldraw.org/legal-info) for details.
