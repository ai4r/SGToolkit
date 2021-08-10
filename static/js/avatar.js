class Avatar {
    constructor(canvas) {
        this.canvas = canvas;
        this.scene = null;
        this.keypointsCursor = -1;

        // meanVec is in Babylonjs coordinate (y values are inverted)
        this.meanVec = [-0.00225, 0.98496, 0.16212, // spine
            0.01831, 0.79641, 0.52568, // head
            0.02496, 0.65216, -0.67807,
            -0.87815, -0.40211, -0.06526, // left arm
            -0.38831, -0.85245, 0.13283,
            0.35888, 0.16606, 0.70720,
            0.87728, -0.41491, -0.00166, // right arm
            0.38441, -0.85739, 0.14593,
            -0.39277, 0.17973, 0.69081];


        this.mixamoRigGestureMapping = {
            "mixamorig:Spine": 0,
            "mixamorig:Head": 1,
            // "mixamorig:HeadTop_End": 2,
            "mixamorig:LeftShoulder": 3,
            "mixamorig:LeftArm": 4,
            "mixamorig:LeftForeArm": 5,
            "mixamorig:RightShoulder": 6,
            "mixamorig:RightArm": 7,
            "mixamorig:RightForeArm": 8,
        };
        this.mixamoPositionGizmoAttachBones = {//"mixamorig:Spine": 0,
            "mixamorig:Head": 1,
            // "mixamorig:HeadTop_End": 2,
            //"mixamorig:LeftShoulder": 3,
            "mixamorig:LeftArm": 4,
            "mixamorig:LeftForeArm": 5,
            "mixamorig:LeftHand": 12,
            //"mixamorig:RightShoulder": 6,
            "mixamorig:RightArm": 7,
            "mixamorig:RightForeArm": 8,
            "mixamorig:RightHand": 36,
        };
        this.mixamoRotationGizmoAttachBones = {
            "mixamorig:Spine": 0,
            "mixamorig:Head": 1,
            // "mixamorig:HeadTop_End": 2,
            //"mixamorig:LeftShoulder": 3,
            "mixamorig:LeftArm": 4,
            "mixamorig:LeftForeArm": 5,
            //"mixamorig:RightShoulder": 6,
            "mixamorig:RightArm": 7,
            "mixamorig:RightForeArm": 8,
        };
        this.mixamoRigHandInitialRotation = {
            "mixamorig:RightHandThumb2": [0.0, 0.0, 0.5],
            "mixamorig:RightHandThumb3": [0.0, 0.0, 0.3],
            "mixamorig:RightHandIndex1": [0.2, -0.3],
            "mixamorig:RightHandIndex2": [0.2, 0],
            "mixamorig:RightHandMiddle1": [0.2, 0],
            "mixamorig:RightHandMiddle2": [0.2, 0],
            "mixamorig:RightHandRing1": [0.2, 0.3],
            "mixamorig:RightHandRing2": [0.2, 0],
            "mixamorig:RightHandPinky1": [0.2, 0.5],
            "mixamorig:RightHandPinky2": [0.2, 0],
            "mixamorig:LeftHandThumb2": [0.0, 0.0, -0.5],
            "mixamorig:LeftHandThumb3": [0.0, 0.0, -0.3],
            "mixamorig:LeftHandIndex1": [0.2, 0.3],
            "mixamorig:LeftHandIndex2": [0.2, 0],
            "mixamorig:LeftHandMiddle1": [0.2, 0],
            "mixamorig:LeftHandMiddle2": [0.2, 0],
            "mixamorig:LeftHandRing1": [0.2, -0.3],
            "mixamorig:LeftHandRing2": [0.2, 0],
            "mixamorig:LeftHandPinky1": [0.2, -0.5],
            "mixamorig:LeftHandPinky2": [0.2, 0],
        }
        //this.gestureKeypoints = null;
        this.setupBabylon(canvas);
    }

    setupBabylon(canvas) {
        var that = this
        var engine = new BABYLON.Engine(canvas, true);
        var scene = new BABYLON.Scene(engine);

        var camera = new BABYLON.ArcRotateCamera("camera", 2, 1.3, 1.2, BABYLON.Vector3.Zero(), scene);
        camera.setTarget(new BABYLON.Vector3(0, 0, 0));
        camera.attachControl(canvas, true);
        camera.wheelPrecision = 100;
        camera.minZ = 0;

        var light = new BABYLON.HemisphericLight("light1", new BABYLON.Vector3(0, 1, 0), scene);
        light.intensity = 1.0;

        /******* Add the create scene function ******/
        var createScene = function () {
            BABYLON.SceneLoader.ImportMesh("", "mesh/mannequin/", "mannequin.babylon", scene, function (newMeshes, particleSystems, skeletons) {
                // prevent mesh disappear
                for (var elem of newMeshes) {
                    elem.alwaysSelectAsActiveMesh = true
                }

                // background
                var helper = scene.createDefaultEnvironment({
                    enableGroundShadow: true
                });
                helper.setMainColor(BABYLON.Color3.Gray());
                helper.ground.position.y -= 1.3;

                // model should follow mixamo rig structure
                var mesh = newMeshes[0];
                var skeleton = skeletons[0];

                function moveSkeleton(skelton, vec, mesh) {
                    var root = skelton.bones[0];
                    root.translate(vec, BABYLON.Space.WORLD, mesh);
                }

                mesh.scaling = new BABYLON.Vector3(1, 1, 1);
                // mesh.translate(BABYLON.Axis.Y, -1.5, BABYLON.Space.WORLD);
                // this moves mesh too.
                moveSkeleton(skeleton, new BABYLON.Vector3(0, -1.3, 0), mesh);

                function createAxes(scene) {
                    for (var i = 0; i < skeleton.bones.length; i++) {
                        new BABYLON.Debug.BoneAxesViewer(scene, skeleton.bones[i], mesh, .5).update();
                    }
                }

                //createAxes();

                // set default pose
                that.restPose()

                // set default hand pose
                for (var i = 0; i < skeleton.bones.length; i++) {
                    var boneName = skeleton.bones[i].id
                    if (boneName in that.mixamoRigHandInitialRotation) {
                        var rotVal = that.mixamoRigHandInitialRotation[boneName];
                        skeleton.bones[i].rotate(BABYLON.Axis.X, rotVal[0])
                        skeleton.bones[i].rotate(BABYLON.Axis.Y, rotVal[1])
                        if (rotVal.length == 3) {
                            skeleton.bones[i].rotate(BABYLON.Axis.Z, rotVal[2])
                        }
                    }
                }

            });
            return scene;
        };
        /******* End of the create scene function ******/

        scene = createScene(); //Call the createScene function

        // Register a render loop to repeatedly render the scene
        engine.runRenderLoop(function () {
            scene.render();
        });

        // Watch for browser/canvas resize events
        window.addEventListener("resize", function () {
            engine.resize();
        });

        this.scene = scene;
    }

    resetCursor() {
        this.keypointsCursor = -1;
    }

    refinePose(boneName, vec, allJoints) {
        // NOTE: nothing to refine yet
        return vec
    }

    moveBody(pose) {
        // only single skeleton and meshes in the scene
        var mesh = this.getMesh();
        var bones = this.getBones();

        if (pose == null || bones == null) {
            return;
        }

        for (var i = 0; i < bones.length; i++) {
            var boneName = bones[i]["name"];

            if (boneName in this.mixamoRigGestureMapping) {
                var keypointIdx = this.mixamoRigGestureMapping[boneName];
                var vecX = pose[keypointIdx * 3];
                var vecY = pose[keypointIdx * 3 + 1];
                var vecZ = pose[keypointIdx * 3 + 2];

                var targetVec = new BABYLON.Vector3(vecX, vecY, vecZ);
                targetVec = this.refinePose(boneName, targetVec, pose)

                this.moveBone(bones[i], targetVec);
            }

            if (boneName == "mixamorig:Spine") {
                // set spine yaw according to the shoulder angle

                // target rotation
                let lsIdx = this.mixamoRigGestureMapping["mixamorig:LeftShoulder"];
                let rsIdx = this.mixamoRigGestureMapping["mixamorig:RightShoulder"];
                let ls = new BABYLON.Vector3(pose[lsIdx * 3], pose[lsIdx * 3 + 1], pose[lsIdx * 3 + 2]);
                let rs = new BABYLON.Vector3(pose[rsIdx * 3], pose[rsIdx * 3 + 1], pose[rsIdx * 3 + 2]);
                let targetRotation = -BABYLON.Vector3.GetAngleBetweenVectors(new BABYLON.Vector3(1, 0, 0), rs.subtract(ls), new BABYLON.Vector3(0, 1, 0))

                // current rotation
                let curRotation = bones[i].getRotation()
                let rotVal = curRotation.y - targetRotation

                bones[i].rotate(new BABYLON.Vector3(0, 1, 0), rotVal, BABYLON.Space.WORLD, mesh);
            } else if (boneName == "mixamorig:Head") {
                // set head yaw according to neck-head vector

                let idx = this.mixamoRigGestureMapping["mixamorig:Head"];
                let headVec = new BABYLON.Vector3(pose[idx * 3], pose[idx * 3 + 1], pose[idx * 3 + 2]);

                var rotResult = this.getRotation(new BABYLON.Vector3(0, 1, 0), headVec)
                var rotAxis = rotResult[0];
                var rotAngle = rotResult[1];
                var euler = BABYLON.Quaternion.RotationAxis(rotAxis, rotAngle).toEulerAngles();

                euler.x = euler.x - 0.3  // head up a little bit
                euler.y = -euler.z
                euler.z = 0  // roll
                bones[i].setRotation(euler, BABYLON.Space.WORLD, mesh)
            }
        }
    }

    restPose() {
        var bones = this.getBones();

        if (bones == null) {
            return;
        }

        // set default rotations
        for (var i = 0; i < bones.length; i++) {
            var boneName = bones[i]["name"];
            // if(boneName in this.mixamoRigGestureMapping) {
            if (boneName.includes("Arm") || boneName.includes("Spine") || boneName.includes("Head")) {
                let rot = new BABYLON.Vector3(0, 0, 0);
                bones[i].setRotation(rot, BABYLON.Space.LOCAL);
            }
        }

        this.moveBody(this.meanVec)
    }

    isBonesNaN() {
        var mesh = this.getMesh();
        var bones = this.getBones();
        var i = 0;
        for (i = 0; i < bones.length; i++) {
            var boneName = bones[i]["name"];
            if (boneName in this.mixamoRigGestureMapping) {
                var b = bones[i];
                var p = b.getAbsolutePosition(mesh);
                var l = p.length();
                if (isNaN(l)) {
                    return true;
                }
            }
        }
        return false;
    }

    getBones() {
        var skeleton = this.getSkeleton();
        return skeleton.bones;
    }

    getControllableBones() {
        var bones = this.getBones();
        var controllables = [];
        for (var i = 0; i < bones.length; i++) {
            var bone = bones[i];
            var boneName = bone["name"];
            if (boneName in this.mixamoRigGestureMapping) {
                controllables.push(bone);
            }
        }

        return controllables;
    }

    getPositionGizmoAttachableBones() {
        var bones = this.getBones();
        var bonesToAttach = [];
        for (var i = 0; i < bones.length; i++) {
            var bone = bones[i];
            var boneName = bone["name"];
            if (boneName in this.mixamoPositionGizmoAttachBones) {
                bonesToAttach.push(bone);
            }
        }
        return bonesToAttach;
    }

    getRotationGizmoAttachbleBones() {
        var bones = this.getBones();
        var bonesToAttach = [];
        for (var i = 0; i < bones.length; i++) {
            var bone = bones[i];
            var boneName = bone["name"];
            if (boneName in this.mixamoRotationGizmoAttachBones) {
                bonesToAttach.push(bone);
            }
        }
        return bonesToAttach;
    }

    getSkeleton() {
        return this.scene.skeletons[0];
    }

    getMesh() {
        return this.scene.meshes[0];
    }

    moveBone(bone, targetVec) {
        var boneDirection = this.getBoneDirection(bone);
        this.moveBoneAsVecRotation(bone, boneDirection, targetVec);
    }

    moveBoneAsVecRotation(bone, fromVec, toVec) {
        var mesh = this.getMesh();

        var rotResult = this.getRotation(fromVec, toVec);
        var rotAxis = rotResult[0];
        var rotAngle = rotResult[1];
        var boneScale = bone.getScale();
        bone.rotate(rotAxis, rotAngle, BABYLON.Space.WORLD, mesh);
        // I don't know why but the scale of the bone slightly changes when it is rotated.
        // foring the bone to maintain its scale.
        bone.setScale(new BABYLON.Vector3(boneScale.x, boneScale.y, boneScale.z));
    }

    getBonePosition(bone) {
        var mesh = this.getMesh();
        return bone.getAbsolutePosition(mesh);
    }

    setBonePosition(bone, pos) {
        var mesh = this.getMesh();
        return bone.setAbsolutePosition(pos, mesh);
    }

    getBoneDirection(bone) {
        var mesh = this.getMesh();
        var childBones = bone.getChildren();
        if (childBones.length == 0) {
            return new BABYLON.Vector3(0, 1, 0);
        }

        var childBone = childBones[0];
        var childBonePose = this.getBonePosition(childBone);
        var thisBonePose = this.getBonePosition(bone);
        var boneDirection = childBonePose.subtract(thisBonePose);
        boneDirection = BABYLON.Vector3.Normalize(boneDirection);

        return boneDirection;

    }

    drawBoneDirections() {
        var bones = this.getControllableBones();
        for (var i = 0; i < bones.length; i++) {
            this.drawBoneDirection(bones[i]);
        }
    }

    drawBoneDirection(bone) {
        var bonePosition = this.getBonePosition(bone);
        var boneDirection = this.getBoneDirection(bone);

        var dispVec = boneDirection;
        dispVec = bonePosition.add(dispVec)
        var axis = BABYLON.Mesh.CreateLines("axis", [bonePosition, dispVec], this.scene);
        axis.color = new BABYLON.Color3(0, 0, 1);
    }

    getRotation(fromVec, toVec) {
        var fromNorm = BABYLON.Vector3.Normalize(fromVec);
        var toNorm = BABYLON.Vector3.Normalize(toVec);
        var dot = BABYLON.Vector3.Dot(fromNorm, toNorm);

        // handle float operation error in vector normalization.
        if (dot > 1.0) {
            dot = 1.0;
        }

        var angle = Math.acos(dot);
        var axis = BABYLON.Vector3.Cross(fromNorm, toNorm);
        var quaternion = BABYLON.Quaternion.RotationAxis(axis, angle);
        return [axis, angle];
    }

    /*
    convertToBabylonCoord(vec) {
      var converted = new BABYLON.Vector3(vec.x, -vec.y, vec.z);
      return converted;
    }

    convertToGeneratorCoord(vec) {
      var converted = new BABYLON.Vector3(vec.x, -vec.y, vec.z);
      return converted;
    }
    */
    emptyArrayAs(data) {
        var newData = [];
        for (var i = 0; i < data.length; i++) {
            newData.push(0);
        }
        return newData;
    }

    copyArray(data) {
        var newData = [];
        for (var i = 0; i < data.length; i++) {
            newData.push(data[i]);
        }
        return newData;
    }

    modelKeypointsToArray() {
        var bones = this.getControllableBones();
        var data = [];

        for (var i = 0; i < bones.length; i++) {
            data.push(0);
            data.push(0);
            data.push(0);
        }

        for (var i = 0; i < bones.length; i++) {
            var bone = bones[i];
            var name = bone['name'];

            var direction = this.getBoneDirection(bone);
            //var convertedDir = this.convertToGeneratorCoord(direction);

            var idx = this.mixamoRigGestureMapping[name];
            data[idx * 3] = direction.x;
            data[idx * 3 + 1] = direction.y;
            data[idx * 3 + 2] = direction.z;
        }

        return data;
    }

    registerForManualTranslation(bone) {
        var parentBone = bone.getParent();
        if (parentBone == null) {
            return;
        }
        this.beforeTraslationDirection = this.getBoneDirection(parentBone);
        this.boneForManualTranslation = bone;
    }

    refineBoneAfterManualTraslation(bone) {

        if (this.boneForManualTranslation == null) {
            return;
        }
        if (bone.name != this.boneForManualTranslation.name) {
            throw "input bone is not registered";
        }

        this.connectBoneWithParent(bone)

    }

    connectBoneWithParent(bone) {

        /*
          - bone is disconnected with parent because the bone is traslated by hand.
          - When connecting, update direction of the parent bone with new bone pos.
          - This function should not use getBoneDirection, because getBoneDirecton uses
            child node. But in this case, child node is alread moved, so it gives wrong direction.
        */

        var parentBone = bone.getParent();

        var newDirection = this.getBoneDirection(parentBone);

        this.moveBoneAsVecRotation(parentBone, this.beforeTraslationDirection, newDirection);
        var parentPos = this.getBonePosition(parentBone);
        var endOfParentBone = parentPos.add(newDirection.scale(parentBone.length));
        this.setBonePosition(bone, endOfParentBone);
    }
}
