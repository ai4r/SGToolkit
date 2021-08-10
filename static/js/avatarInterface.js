/*
Handles mouse interaction with avatar.
Joint controls are done in here.
This should not hold an avatar object.
*/
class AvatarInterface {
    constructor(avatar) {
        this.attachInterationCallback(avatar);
        //this.handleMeshes = this.attachControlHandles(avatar);
        this.skeletonViewer = this.addSkeletonViewer(avatar);
        this.rotationGizmos = this.createRotationGizmo(avatar);
        this.positionGizmos = this.createPositionGizmo(avatar);

        this.currentGizmoType = 'none';
        this.turnOffEditMode();

        this.gizmoDragEndCallback = null;
        this.gizmoDragCallback = null;
    }

    attachControlHandles(avatar) {
        // attach invisible spheres to control joints directly
        var bones = avatar.getControllableBones();
        var avatarMesh = avatar.getMesh();
        var handleMeshes = [];
        for (var i = 0; i < bones.length; i++) {
            var sphere = BABYLON.MeshBuilder.CreateSphere("sphere", {diameter: 0.1}, avatar.scene); //default sphere
            var bone = bones[i];
            sphere.attachToBone(bone, avatarMesh);
            handleMeshes.push(sphere);
            sphere.setEnabled(false);
        }
        return handleMeshes;
    }

    addSkeletonViewer(avatar) {
        var skeleton = avatar.getSkeleton();
        var mesh = avatar.getMesh();
        var scene = avatar.scene;
        var skeletonViewer = new BABYLON.Debug.SkeletonViewer(skeleton, mesh, scene);
        skeletonViewer.isEnabled = true;
        skeletonViewer.color = BABYLON.Color3.Yellow(); // Change default color from white to red
        return skeletonViewer;
    }

    attachInterationCallback(avatar) {
        avatar.scene.onPointerObservable.add(function (pointerInfo) {
            switch (pointerInfo.type) {
                case BABYLON.PointerEventTypes.POINTERDOWN:
                    this.pointerDown(pointerInfo);
                case BABYLON.PointerEventTypes.POINTERMOVE:
                    this.pointerMove(pointerInfo);

            }
        }.bind(this));
    }

    pointerDown(pointerInfo) {

    }

    pointerUp() {
        //console.log("pointer up");
    }

    pointerMove() {
        //console.log("pointer move");
    }

    turnOnEditMode(avatar, gizmoType) {
        this.turnOffEditMode();

        this.skeletonViewer.isEnabled = true;

        if (gizmoType == 'rotation') {
            this.attachRotationGizmo(avatar);
        } else if (gizmoType == 'position') {
            this.attachPostionGizmo(avatar);
        }

        this.currentGizmoType = gizmoType;

    }

    turnOffEditMode() {
        this.skeletonViewer.isEnabled = false;
        if (this.currentGizmoType == 'none') {
            return;
        }

        if (this.currentGizmoType == 'rotation') {
            this.detachGizmos(this.rotationGizmos);
        } else if (this.currentGizmoType == 'position') {
            this.detachGizmos(this.positionGizmos);
        }

    }

    createPositionGizmo(avatar) {
        var gizmos = [];
        var gizmoScale = 0.75
        var utilLayer = new BABYLON.UtilityLayerRenderer(avatar.scene);
        var that = this;

        function getOnDragEndCallback(bone) {
            // You have to do this weired thing to freeze (or save)
            // variable in the loop for callback
            // https://stackoverflow.com/questions/7053965/when-using-callbacks-inside-a-loop-in-javascript-is-there-any-way-to-save-a-var
            return function () {
                if (that.gizmoDragEndCallback != null) {
                    that.gizmoDragEndCallback();
                }
                avatar.refineBoneAfterManualTraslation(bone);
            }
        }

        function getOnDragStartCallback(bone) {
            return function () {
                avatar.registerForManualTranslation(bone);
            }
        }

        var bonesToAttach = avatar.getPositionGizmoAttachableBones();

        for (var i = 0; i < bonesToAttach.length; i++) {
            var gizmo = new BABYLON.PositionGizmo(utilLayer);
            var bone = bonesToAttach[i];
            gizmo.onDragStartObservable.add(getOnDragStartCallback(bone));
            gizmo.onDragEndObservable.add(getOnDragEndCallback(bone));
            gizmo.scaleRatio = gizmoScale;
            gizmos.push(gizmo);
        }

        return gizmos;
    }

    createRotationGizmo(avatar) {

        var gizmos = [];
        var gizmoScale = 0.75
        var utilLayer = new BABYLON.UtilityLayerRenderer(avatar.scene);
        var that = this;

        var bonesToAttach = avatar.getRotationGizmoAttachbleBones();

        for (var i = 0; i < bonesToAttach.length; i++) {
            var gizmo = new BABYLON.RotationGizmo(utilLayer);
            var bone = bonesToAttach[i];
            gizmo.onDragEndObservable.add(function () {
                if (that.gizmoDragEndCallback != null) {
                    that.gizmoDragEndCallback();
                }
            });
            gizmo.scaleRatio = gizmoScale;
            gizmos.push(gizmo);
        }

        return gizmos;

    }

    createGizmos(avatar, bonesToAttach, gizmoClass) {

    }

    attachPostionGizmo(avatar) {
        this.attachGizmos(avatar, avatar.getPositionGizmoAttachableBones(), this.positionGizmos);
    }

    attachRotationGizmo(avatar) {
        this.attachGizmos(avatar, avatar.getRotationGizmoAttachbleBones(), this.rotationGizmos);
    }

    attachGizmos(avatar, bonesToAttach, gizmos) {
        for (var i = 0; i < bonesToAttach.length; i++) {
            var bone = bonesToAttach[i];
            var gizmo = gizmos[i];
            gizmo.attachedNode = bone;
        }
    }

    detachGizmos(gizmos) {
        for (var i = 0; i < gizmos.length; i++) {
            var gizmo = gizmos[i];
            gizmo.attachedNode = null;
        }
    }
}
