/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

#if defined(HK_COMPILER_SNC)
#	pragma diag_suppress=68 //integer sign change
#endif

namespace hkHavok610r1Classes
{
	const char VersionString[] = "Havok-6.1.0-r1";
	const int ClassVersion = 6;

	extern hkClass hclActionClass;
	extern hkClass hclBendLinkConstraintSetClass;
	extern hkClass hclBendLinkConstraintSetLinkClass;
	extern hkClass hclBlendSomeVerticesOperatorBlendEntryClass;
	extern hkClass hclBlendSomeVerticesOperatorClass;
	extern hkClass hclBonePlanesConstraintSetBonePlaneClass;
	extern hkClass hclBonePlanesConstraintSetClass;
	extern hkClass hclBufferDefinitionClass;
	extern hkClass hclCapsuleShapeClass;
	extern hkClass hclClothContainerClass;
	extern hkClass hclClothDataClass;
	extern hkClass hclClothStateBufferAccessClass;
	extern hkClass hclClothStateClass;
	extern hkClass hclCollidableClass;
	extern hkClass hclConstraintSetClass;
	extern hkClass hclConvexHeightFieldShapeClass;
	extern hkClass hclCopyVerticesOperatorClass;
	extern hkClass hclCylinderShapeClass;
	extern hkClass hclGatherAllVerticesOperatorClass;
	extern hkClass hclGatherSomeVerticesOperatorClass;
	extern hkClass hclGatherSomeVerticesOperatorVertexPairClass;
	extern hkClass hclHingeConstraintSetClass;
	extern hkClass hclHingeConstraintSetHingeClass;
	extern hkClass hclLocalRangeConstraintSetClass;
	extern hkClass hclLocalRangeConstraintSetLocalConstraintClass;
	extern hkClass hclMeshMeshDeformOperatorClass;
	extern hkClass hclMeshMeshDeformOperatorTriangleVertexPairClass;
	extern hkClass hclMoveParticlesOperatorClass;
	extern hkClass hclMoveParticlesOperatorVertexParticlePairClass;
	extern hkClass hclOperatorClass;
	extern hkClass hclPlaneShapeClass;
	extern hkClass hclRecalculateAllNormalsOperatorClass;
	extern hkClass hclRecalculateSomeNormalsOperatorClass;
	extern hkClass hclShapeClass;
	extern hkClass hclSimClothDataClass;
	extern hkClass hclSimClothDataCollisionPairClass;
	extern hkClass hclSimClothDataParticleDataClass;
	extern hkClass hclSimClothDataSimulationInfoClass;
	extern hkClass hclSimClothPoseClass;
	extern hkClass hclSimpleWindActionClass;
	extern hkClass hclSimulateOperatorClass;
	extern hkClass hclSkinOperatorBoneInfluenceClass;
	extern hkClass hclSkinOperatorClass;
	extern hkClass hclSphereShapeClass;
	extern hkClass hclStandardLinkConstraintSetClass;
	extern hkClass hclStandardLinkConstraintSetLinkClass;
	extern hkClass hclStretchLinkConstraintSetClass;
	extern hkClass hclStretchLinkConstraintSetLinkClass;
	extern hkClass hclToolNamedObjectReferenceClass;
	extern hkClass hclTransformSetDefinitionClass;
	extern hkClass hclTransitionConstraintSetClass;
	extern hkClass hclTransitionConstraintSetPerParticleClass;
	extern hkClass hclTriangleSelectionInputClass;
	extern hkClass hclVertexFloatInputClass;
	extern hkClass hclVertexSelectionInputClass;
	extern hkClass hclVolumeConstraintApplyDataClass;
	extern hkClass hclVolumeConstraintClass;
	extern hkClass hclVolumeConstraintFrameDataClass;
	extern hkClass hkAabbClass;
	extern hkClass hkAabbUint32Class;
	extern hkClass hkBaseObjectClass;
	extern hkClass hkBitFieldClass;
	extern hkClass hkClassClass;
	extern hkClass hkClassEnumClass;
	extern hkClass hkClassEnumItemClass;
	extern hkClass hkClassMemberClass;
	extern hkClass hkColorClass;
	extern hkClass hkContactPointClass;
	extern hkClass hkContactPointMaterialClass;
	extern hkClass hkCustomAttributesAttributeClass;
	extern hkClass hkCustomAttributesClass;
	extern hkClass hkLocalFrameClass;
	extern hkClass hkMemoryResourceContainerClass;
	extern hkClass hkMemoryResourceHandleClass;
	extern hkClass hkMemoryResourceHandleExternalLinkClass;
	extern hkClass hkMonitorStreamColorTableClass;
	extern hkClass hkMonitorStreamColorTableColorPairClass;
	extern hkClass hkMonitorStreamFrameInfoClass;
	extern hkClass hkMonitorStreamStringMapClass;
	extern hkClass hkMonitorStreamStringMapStringMapClass;
	extern hkClass hkMoppBvTreeShapeBaseClass;
	extern hkClass hkMotionStateClass;
	extern hkClass hkMultiThreadCheckClass;
	extern hkClass hkPackfileHeaderClass;
	extern hkClass hkPackfileSectionHeaderClass;
	extern hkClass hkReferencedObjectClass;
	extern hkClass hkResourceBaseClass;
	extern hkClass hkResourceContainerClass;
	extern hkClass hkResourceHandleClass;
	extern hkClass hkRootLevelContainerClass;
	extern hkClass hkRootLevelContainerNamedVariantClass;
	extern hkClass hkSimpleLocalFrameClass;
	extern hkClass hkSphereClass;
	extern hkClass hkSweptTransformClass;
	extern hkClass hkVariableTweakingHelperBoolVariableInfoClass;
	extern hkClass hkVariableTweakingHelperClass;
	extern hkClass hkVariableTweakingHelperIntVariableInfoClass;
	extern hkClass hkVariableTweakingHelperRealVariableInfoClass;
	extern hkClass hkVariableTweakingHelperVector4VariableInfoClass;
	extern hkClass hkWorldMemoryAvailableWatchDogClass;
	extern hkClass hkaAnimatedReferenceFrameClass;
	extern hkClass hkaAnimationBindingClass;
	extern hkClass hkaAnimationClass;
	extern hkClass hkaAnimationContainerClass;
	extern hkClass hkaAnimationPreviewColorClass;
	extern hkClass hkaAnimationPreviewColorContainerClass;
	extern hkClass hkaAnnotationTrackAnnotationClass;
	extern hkClass hkaAnnotationTrackClass;
	extern hkClass hkaBoneAttachmentClass;
	extern hkClass hkaBoneClass;
	extern hkClass hkaDefaultAnimatedReferenceFrameClass;
	extern hkClass hkaDeltaCompressedAnimationClass;
	extern hkClass hkaDeltaCompressedAnimationQuantizationFormatClass;
	extern hkClass hkaFootstepAnalysisInfoClass;
	extern hkClass hkaFootstepAnalysisInfoContainerClass;
	extern hkClass hkaInterleavedUncompressedAnimationClass;
	extern hkClass hkaKeyFrameHierarchyUtilityClass;
	extern hkClass hkaKeyFrameHierarchyUtilityControlDataClass;
	extern hkClass hkaMeshBindingClass;
	extern hkClass hkaMeshBindingMappingClass;
	extern hkClass hkaRagdollInstanceClass;
	extern hkClass hkaSkeletonClass;
	extern hkClass hkaSkeletonLocalFrameOnBoneClass;
	extern hkClass hkaSkeletonMapperClass;
	extern hkClass hkaSkeletonMapperDataChainMappingClass;
	extern hkClass hkaSkeletonMapperDataClass;
	extern hkClass hkaSkeletonMapperDataSimpleMappingClass;
	extern hkClass hkaSplineCompressedAnimationAnimationCompressionParamsClass;
	extern hkClass hkaSplineCompressedAnimationClass;
	extern hkClass hkaSplineCompressedAnimationTrackCompressionParamsClass;
	extern hkClass hkaWaveletCompressedAnimationClass;
	extern hkClass hkaWaveletCompressedAnimationCompressionParamsClass;
	extern hkClass hkaWaveletCompressedAnimationQuantizationFormatClass;
	extern hkClass hkbAlignBoneModifierClass;
	extern hkClass hkbAnimatedSkeletonGeneratorClass;
	extern hkClass hkbAttachmentModifierClass;
	extern hkClass hkbAttachmentSetupClass;
	extern hkClass hkbAttributeModifierAssignmentClass;
	extern hkClass hkbAttributeModifierClass;
	extern hkClass hkbBalanceControllerModifierClass;
	extern hkClass hkbBalanceModifierClass;
	extern hkClass hkbBalanceModifierStepInfoClass;
	extern hkClass hkbBalanceRadialSelectorGeneratorClass;
	extern hkClass hkbBehaviorGraphClass;
	extern hkClass hkbBehaviorGraphDataClass;
	extern hkClass hkbBehaviorGraphStringDataClass;
	extern hkClass hkbBehaviorReferenceGeneratorClass;
	extern hkClass hkbBlendCurveUtilsClass;
	extern hkClass hkbBlenderGeneratorChildClass;
	extern hkClass hkbBlenderGeneratorClass;
	extern hkClass hkbBlendingTransitionEffectClass;
	extern hkClass hkbBoolVariableSequencedDataClass;
	extern hkClass hkbBoolVariableSequencedDataSampleClass;
	extern hkClass hkbCatchFallModifierClass;
	extern hkClass hkbCatchFallModifierHandClass;
	extern hkClass hkbCharacterBoneInfoClass;
	extern hkClass hkbCharacterClass;
	extern hkClass hkbCharacterDataClass;
	extern hkClass hkbCharacterSetupClass;
	extern hkClass hkbCharacterStringDataClass;
	extern hkClass hkbCheckBalanceModifierClass;
	extern hkClass hkbCheckRagdollSpeedModifierClass;
	extern hkClass hkbClimbMountingPredicateClass;
	extern hkClass hkbClipGeneratorClass;
	extern hkClass hkbClipTriggerClass;
	extern hkClass hkbComputeDirectionModifierClass;
	extern hkClass hkbComputeWorldFromModelModifierClass;
	extern hkClass hkbConstrainRigidBodyModifierClass;
	extern hkClass hkbContextClass;
	extern hkClass hkbControlledReachModifierClass;
	extern hkClass hkbCustomTestGeneratorClass;
	extern hkClass hkbCustomTestGeneratorStruckClass;
	extern hkClass hkbDampingModifierClass;
	extern hkClass hkbDelayedModifierClass;
	extern hkClass hkbDemoConfigCharacterInfoClass;
	extern hkClass hkbDemoConfigClass;
	extern hkClass hkbDemoConfigStickVariableInfoClass;
	extern hkClass hkbDemoConfigTerrainInfoClass;
	extern hkClass hkbDetectCloseToGroundModifierClass;
	extern hkClass hkbEvaluateHandleModifierClass;
	extern hkClass hkbEventClass;
	extern hkClass hkbEventDrivenModifierClass;
	extern hkClass hkbEventInfoClass;
	extern hkClass hkbEventSequencedDataClass;
	extern hkClass hkbEventSequencedDataSequencedEventClass;
	extern hkClass hkbExtrapolatingTransitionEffectClass;
	extern hkClass hkbFaceTargetModifierClass;
	extern hkClass hkbFootIkControlDataClass;
	extern hkClass hkbFootIkControlsModifierClass;
	extern hkClass hkbFootIkGainsClass;
	extern hkClass hkbFootIkModifierClass;
	extern hkClass hkbFootIkModifierInternalLegDataClass;
	extern hkClass hkbFootIkModifierLegClass;
	extern hkClass hkbGeneratorClass;
	extern hkClass hkbGeneratorOutputClass;
	extern hkClass hkbGeneratorOutputConstTrackClass;
	extern hkClass hkbGeneratorOutputListenerClass;
	extern hkClass hkbGeneratorOutputTrackClass;
	extern hkClass hkbGeneratorOutputTrackHeaderClass;
	extern hkClass hkbGeneratorOutputTrackMasterHeaderClass;
	extern hkClass hkbGeneratorOutputTracksClass;
	extern hkClass hkbGeneratorTransitionEffectClass;
	extern hkClass hkbGetHandleOnBoneModifierClass;
	extern hkClass hkbGetUpModifierClass;
	extern hkClass hkbGravityModifierClass;
	extern hkClass hkbHandIkControlDataClass;
	extern hkClass hkbHandIkControlsModifierClass;
	extern hkClass hkbHandIkControlsModifierHandClass;
	extern hkClass hkbHandIkModifierClass;
	extern hkClass hkbHandIkModifierHandClass;
	extern hkClass hkbHandleClass;
	extern hkClass hkbHoldFromBlendingTransitionEffectClass;
	extern hkClass hkbIntVariableSequencedDataClass;
	extern hkClass hkbIntVariableSequencedDataSampleClass;
	extern hkClass hkbJigglerGroupClass;
	extern hkClass hkbJigglerModifierClass;
	extern hkClass hkbKeyframeBonesModifierClass;
	extern hkClass hkbLookAtModifierClass;
	extern hkClass hkbManualSelectorGeneratorClass;
	extern hkClass hkbMirrorModifierClass;
	extern hkClass hkbMirroredSkeletonInfoClass;
	extern hkClass hkbModifierClass;
	extern hkClass hkbModifierGeneratorClass;
	extern hkClass hkbModifierListClass;
	extern hkClass hkbModifierWrapperClass;
	extern hkClass hkbMoveBoneTowardTargetModifierClass;
	extern hkClass hkbMoveCharacterModifierClass;
	extern hkClass hkbNodeClass;
	extern hkClass hkbPoseMatchingGeneratorClass;
	extern hkClass hkbPoseStoringGeneratorOutputListenerClass;
	extern hkClass hkbPoseStoringGeneratorOutputListenerStoredPoseClass;
	extern hkClass hkbPositionRelativeSelectorGeneratorClass;
	extern hkClass hkbPoweredRagdollControlDataClass;
	extern hkClass hkbPoweredRagdollControlsModifierClass;
	extern hkClass hkbPoweredRagdollModifierClass;
	extern hkClass hkbPoweredRagdollModifierKeyframeInfoClass;
	extern hkClass hkbPredicateClass;
	extern hkClass hkbProjectDataClass;
	extern hkClass hkbProjectStringDataClass;
	extern hkClass hkbProxyModifierClass;
	extern hkClass hkbProxyModifierProxyInfoClass;
	extern hkClass hkbRadialSelectorGeneratorClass;
	extern hkClass hkbRadialSelectorGeneratorGeneratorInfoClass;
	extern hkClass hkbRadialSelectorGeneratorGeneratorPairClass;
	extern hkClass hkbRagdollDriverModifierClass;
	extern hkClass hkbRagdollForceModifierClass;
	extern hkClass hkbReachModifierClass;
	extern hkClass hkbReachModifierHandClass;
	extern hkClass hkbReachTowardTargetModifierClass;
	extern hkClass hkbReachTowardTargetModifierHandClass;
	extern hkClass hkbRealVariableSequencedDataClass;
	extern hkClass hkbRealVariableSequencedDataSampleClass;
	extern hkClass hkbReferencePoseGeneratorClass;
	extern hkClass hkbRegisteredGeneratorClass;
	extern hkClass hkbRigidBodyRagdollControlDataClass;
	extern hkClass hkbRigidBodyRagdollControlsModifierClass;
	extern hkClass hkbRigidBodyRagdollModifierClass;
	extern hkClass hkbRotateCharacterModifierClass;
	extern hkClass hkbSenseHandleModifierClass;
	extern hkClass hkbSequenceClass;
	extern hkClass hkbSequenceStringDataClass;
	extern hkClass hkbSequencedDataClass;
	extern hkClass hkbSimpleCharacterClass;
	extern hkClass hkbSplinePathGeneratorClass;
	extern hkClass hkbStateDependentModifierClass;
	extern hkClass hkbStateMachineActiveTransitionInfoClass;
	extern hkClass hkbStateMachineClass;
	extern hkClass hkbStateMachineNestedStateMachineDataClass;
	extern hkClass hkbStateMachineProspectiveTransitionInfoClass;
	extern hkClass hkbStateMachineStateInfoClass;
	extern hkClass hkbStateMachineTimeIntervalClass;
	extern hkClass hkbStateMachineTransitionInfoClass;
	extern hkClass hkbStringPredicateClass;
	extern hkClass hkbTargetClass;
	extern hkClass hkbTargetRigidBodyModifierClass;
	extern hkClass hkbTimerModifierClass;
	extern hkClass hkbTransformVectorModifierClass;
	extern hkClass hkbTransitionEffectClass;
	extern hkClass hkbTwistModifierClass;
	extern hkClass hkbVariableBindingSetBindingClass;
	extern hkClass hkbVariableBindingSetClass;
	extern hkClass hkbVariableInfoClass;
	extern hkClass hkbVariableValueClass;
	extern hkClass hkdBallGunBlueprintClass;
	extern hkClass hkdBodyClass;
	extern hkClass hkdBreakableBodyBlueprintClass;
	extern hkClass hkdBreakableBodyClass;
	extern hkClass hkdBreakableBodySmallArraySerializeOverrideTypeClass;
	extern hkClass hkdBreakableShapeClass;
	extern hkClass hkdBreakableShapeConnectionClass;
	extern hkClass hkdChangeMassGunBlueprintClass;
	extern hkClass hkdCompoundBreakableBodyBlueprintClass;
	extern hkClass hkdCompoundBreakableShapeClass;
	extern hkClass hkdContactRegionControllerClass;
	extern hkClass hkdControllerClass;
	extern hkClass hkdDeformableBreakableShapeClass;
	extern hkClass hkdDeformationControllerClass;
	extern hkClass hkdDestructionDemoConfigClass;
	extern hkClass hkdFractureClass;
	extern hkClass hkdGeometryClass;
	extern hkClass hkdGeometryFaceClass;
	extern hkClass hkdGeometryFaceIdentifierClass;
	extern hkClass hkdGeometryObjectIdentifierClass;
	extern hkClass hkdGeometryTriangleClass;
	extern hkClass hkdGravityGunBlueprintClass;
	extern hkClass hkdGrenadeGunBlueprintClass;
	extern hkClass hkdMountedBallGunBlueprintClass;
	extern hkClass hkdPieFractureClass;
	extern hkClass hkdPropertiesClass;
	extern hkClass hkdRandomSplitFractureClass;
	extern hkClass hkdShapeClass;
	extern hkClass hkdShapeInstanceInfoClass;
	extern hkClass hkdShapeInstanceInfoRuntimeInfoClass;
	extern hkClass hkdSliceFractureClass;
	extern hkClass hkdSplitInHalfFractureClass;
	extern hkClass hkdSplitShapeClass;
	extern hkClass hkdWeaponBlueprintClass;
	extern hkClass hkdWoodControllerClass;
	extern hkClass hkdWoodFractureClass;
	extern hkClass hkdWoodFractureSplittingDataClass;
	extern hkClass hkp2dAngConstraintAtomClass;
	extern hkClass hkpAabbPhantomClass;
	extern hkClass hkpActionClass;
	extern hkClass hkpAgent1nSectorClass;
	extern hkClass hkpAngConstraintAtomClass;
	extern hkClass hkpAngFrictionConstraintAtomClass;
	extern hkClass hkpAngLimitConstraintAtomClass;
	extern hkClass hkpAngMotorConstraintAtomClass;
	extern hkClass hkpAngularDashpotActionClass;
	extern hkClass hkpArrayActionClass;
	extern hkClass hkpBallAndSocketConstraintDataAtomsClass;
	extern hkClass hkpBallAndSocketConstraintDataClass;
	extern hkClass hkpBallSocketChainDataClass;
	extern hkClass hkpBallSocketChainDataConstraintInfoClass;
	extern hkClass hkpBallSocketConstraintAtomClass;
	extern hkClass hkpBinaryActionClass;
	extern hkClass hkpBoxMotionClass;
	extern hkClass hkpBoxShapeClass;
	extern hkClass hkpBreakableConstraintDataClass;
	extern hkClass hkpBridgeAtomsClass;
	extern hkClass hkpBridgeConstraintAtomClass;
	extern hkClass hkpBroadPhaseHandleClass;
	extern hkClass hkpBvShapeClass;
	extern hkClass hkpBvTreeShapeClass;
	extern hkClass hkpCachingShapePhantomClass;
	extern hkClass hkpCallbackConstraintMotorClass;
	extern hkClass hkpCapsuleShapeClass;
	extern hkClass hkpCdBodyClass;
	extern hkClass hkpCharacterMotionClass;
	extern hkClass hkpCharacterProxyCinfoClass;
	extern hkClass hkpCollidableBoundingVolumeDataClass;
	extern hkClass hkpCollidableClass;
	extern hkClass hkpCollidableCollidableFilterClass;
	extern hkClass hkpCollisionFilterClass;
	extern hkClass hkpCollisionFilterListClass;
	extern hkClass hkpCompressedSampledHeightFieldShapeClass;
	extern hkClass hkpConeLimitConstraintAtomClass;
	extern hkClass hkpConstrainedSystemFilterClass;
	extern hkClass hkpConstraintAtomClass;
	extern hkClass hkpConstraintChainDataClass;
	extern hkClass hkpConstraintChainInstanceActionClass;
	extern hkClass hkpConstraintChainInstanceClass;
	extern hkClass hkpConstraintDataClass;
	extern hkClass hkpConstraintInstanceClass;
	extern hkClass hkpConstraintMotorClass;
	extern hkClass hkpConvexListFilterClass;
	extern hkClass hkpConvexListShapeClass;
	extern hkClass hkpConvexPieceMeshShapeClass;
	extern hkClass hkpConvexPieceStreamDataClass;
	extern hkClass hkpConvexShapeClass;
	extern hkClass hkpConvexTransformShapeBaseClass;
	extern hkClass hkpConvexTransformShapeClass;
	extern hkClass hkpConvexTranslateShapeClass;
	extern hkClass hkpConvexVerticesConnectivityClass;
	extern hkClass hkpConvexVerticesShapeClass;
	extern hkClass hkpConvexVerticesShapeFourVectorsClass;
	extern hkClass hkpCylinderShapeClass;
	extern hkClass hkpDashpotActionClass;
	extern hkClass hkpDefaultConvexListFilterClass;
	extern hkClass hkpDisableEntityCollisionFilterClass;
	extern hkClass hkpDisplayBindingDataClass;
	extern hkClass hkpEntityClass;
	extern hkClass hkpEntityDeactivatorClass;
	extern hkClass hkpEntityExtendedListenersClass;
	extern hkClass hkpEntitySmallArraySerializeOverrideTypeClass;
	extern hkClass hkpEntitySpuCollisionCallbackClass;
	extern hkClass hkpExtendedMeshShapeClass;
	extern hkClass hkpExtendedMeshShapeShapesSubpartClass;
	extern hkClass hkpExtendedMeshShapeSubpartClass;
	extern hkClass hkpExtendedMeshShapeTrianglesSubpartClass;
	extern hkClass hkpFakeRigidBodyDeactivatorClass;
	extern hkClass hkpFastMeshShapeClass;
	extern hkClass hkpFixedRigidMotionClass;
	extern hkClass hkpGenericConstraintDataClass;
	extern hkClass hkpGenericConstraintDataSchemeClass;
	extern hkClass hkpGenericConstraintDataSchemeConstraintInfoClass;
	extern hkClass hkpGroupCollisionFilterClass;
	extern hkClass hkpGroupFilterClass;
	extern hkClass hkpHeightFieldShapeClass;
	extern hkClass hkpHingeConstraintDataAtomsClass;
	extern hkClass hkpHingeConstraintDataClass;
	extern hkClass hkpHingeLimitsDataAtomsClass;
	extern hkClass hkpHingeLimitsDataClass;
	extern hkClass hkpIgnoreModifierConstraintAtomClass;
	extern hkClass hkpKeyframedRigidMotionClass;
	extern hkClass hkpLimitedForceConstraintMotorClass;
	extern hkClass hkpLimitedHingeConstraintDataAtomsClass;
	extern hkClass hkpLimitedHingeConstraintDataClass;
	extern hkClass hkpLinConstraintAtomClass;
	extern hkClass hkpLinFrictionConstraintAtomClass;
	extern hkClass hkpLinLimitConstraintAtomClass;
	extern hkClass hkpLinMotorConstraintAtomClass;
	extern hkClass hkpLinSoftConstraintAtomClass;
	extern hkClass hkpLinearParametricCurveClass;
	extern hkClass hkpLinkedCollidableClass;
	extern hkClass hkpListShapeChildInfoClass;
	extern hkClass hkpListShapeClass;
	extern hkClass hkpMalleableConstraintDataClass;
	extern hkClass hkpMassChangerModifierConstraintAtomClass;
	extern hkClass hkpMassPropertiesClass;
	extern hkClass hkpMaterialClass;
	extern hkClass hkpMaxSizeMotionClass;
	extern hkClass hkpMeshMaterialClass;
	extern hkClass hkpMeshShapeClass;
	extern hkClass hkpMeshShapeSubpartClass;
	extern hkClass hkpModifierConstraintAtomClass;
	extern hkClass hkpMoppBvTreeShapeClass;
	extern hkClass hkpMoppCodeClass;
	extern hkClass hkpMoppCodeCodeInfoClass;
	extern hkClass hkpMoppCodeReindexedTerminalClass;
	extern hkClass hkpMotionClass;
	extern hkClass hkpMotorActionClass;
	extern hkClass hkpMouseSpringActionClass;
	extern hkClass hkpMovingSurfaceModifierConstraintAtomClass;
	extern hkClass hkpMultiRayShapeClass;
	extern hkClass hkpMultiRayShapeRayClass;
	extern hkClass hkpMultiSphereShapeClass;
	extern hkClass hkpNullCollisionFilterClass;
	extern hkClass hkpOverwritePivotConstraintAtomClass;
	extern hkClass hkpPairwiseCollisionFilterClass;
	extern hkClass hkpPairwiseCollisionFilterCollisionPairClass;
	extern hkClass hkpParametricCurveClass;
	extern hkClass hkpPhantomCallbackShapeClass;
	extern hkClass hkpPhantomClass;
	extern hkClass hkpPhysicsDataClass;
	extern hkClass hkpPhysicsSystemClass;
	extern hkClass hkpPhysicsSystemDisplayBindingClass;
	extern hkClass hkpPhysicsSystemWithContactsClass;
	extern hkClass hkpPlaneShapeClass;
	extern hkClass hkpPointToPathConstraintDataClass;
	extern hkClass hkpPointToPlaneConstraintDataAtomsClass;
	extern hkClass hkpPointToPlaneConstraintDataClass;
	extern hkClass hkpPositionConstraintMotorClass;
	extern hkClass hkpPoweredChainDataClass;
	extern hkClass hkpPoweredChainDataConstraintInfoClass;
	extern hkClass hkpPoweredChainMapperClass;
	extern hkClass hkpPoweredChainMapperLinkInfoClass;
	extern hkClass hkpPoweredChainMapperTargetClass;
	extern hkClass hkpPrismaticConstraintDataAtomsClass;
	extern hkClass hkpPrismaticConstraintDataClass;
	extern hkClass hkpPropertyClass;
	extern hkClass hkpPropertyValueClass;
	extern hkClass hkpPulleyConstraintAtomClass;
	extern hkClass hkpPulleyConstraintDataAtomsClass;
	extern hkClass hkpPulleyConstraintDataClass;
	extern hkClass hkpRagdollConstraintDataAtomsClass;
	extern hkClass hkpRagdollConstraintDataClass;
	extern hkClass hkpRagdollLimitsDataAtomsClass;
	extern hkClass hkpRagdollLimitsDataClass;
	extern hkClass hkpRagdollMotorConstraintAtomClass;
	extern hkClass hkpRayCollidableFilterClass;
	extern hkClass hkpRayShapeCollectionFilterClass;
	extern hkClass hkpRejectRayChassisListenerClass;
	extern hkClass hkpRemoveTerminalsMoppModifierClass;
	extern hkClass hkpReorientActionClass;
	extern hkClass hkpRigidBodyClass;
	extern hkClass hkpRigidBodyDeactivatorClass;
	extern hkClass hkpRigidBodyDisplayBindingClass;
	extern hkClass hkpRotationalConstraintDataAtomsClass;
	extern hkClass hkpRotationalConstraintDataClass;
	extern hkClass hkpSampledHeightFieldShapeClass;
	extern hkClass hkpSerializedAgentNnEntryClass;
	extern hkClass hkpSerializedDisplayMarkerClass;
	extern hkClass hkpSerializedDisplayMarkerListClass;
	extern hkClass hkpSerializedDisplayRbTransformsClass;
	extern hkClass hkpSerializedDisplayRbTransformsDisplayTransformPairClass;
	extern hkClass hkpSerializedSubTrack1nInfoClass;
	extern hkClass hkpSerializedTrack1nInfoClass;
	extern hkClass hkpSetLocalRotationsConstraintAtomClass;
	extern hkClass hkpSetLocalTransformsConstraintAtomClass;
	extern hkClass hkpSetLocalTranslationsConstraintAtomClass;
	extern hkClass hkpShapeClass;
	extern hkClass hkpShapeCollectionClass;
	extern hkClass hkpShapeCollectionFilterClass;
	extern hkClass hkpShapeContainerClass;
	extern hkClass hkpShapeInfoClass;
	extern hkClass hkpShapePhantomClass;
	extern hkClass hkpShapeRayCastInputClass;
	extern hkClass hkpSimpleContactConstraintAtomClass;
	extern hkClass hkpSimpleContactConstraintDataInfoClass;
	extern hkClass hkpSimpleMeshShapeClass;
	extern hkClass hkpSimpleMeshShapeTriangleClass;
	extern hkClass hkpSimpleShapePhantomClass;
	extern hkClass hkpSimpleShapePhantomCollisionDetailClass;
	extern hkClass hkpSingleShapeContainerClass;
	extern hkClass hkpSoftContactModifierConstraintAtomClass;
	extern hkClass hkpSpatialRigidBodyDeactivatorClass;
	extern hkClass hkpSpatialRigidBodyDeactivatorSampleClass;
	extern hkClass hkpSphereMotionClass;
	extern hkClass hkpSphereRepShapeClass;
	extern hkClass hkpSphereShapeClass;
	extern hkClass hkpSpringActionClass;
	extern hkClass hkpSpringDamperConstraintMotorClass;
	extern hkClass hkpStabilizedBoxMotionClass;
	extern hkClass hkpStabilizedSphereMotionClass;
	extern hkClass hkpStiffSpringChainDataClass;
	extern hkClass hkpStiffSpringChainDataConstraintInfoClass;
	extern hkClass hkpStiffSpringConstraintAtomClass;
	extern hkClass hkpStiffSpringConstraintDataAtomsClass;
	extern hkClass hkpStiffSpringConstraintDataClass;
	extern hkClass hkpStorageExtendedMeshShapeClass;
	extern hkClass hkpStorageExtendedMeshShapeMeshSubpartStorageClass;
	extern hkClass hkpStorageExtendedMeshShapeShapeSubpartStorageClass;
	extern hkClass hkpStorageMeshShapeClass;
	extern hkClass hkpStorageMeshShapeSubpartStorageClass;
	extern hkClass hkpStorageSampledHeightFieldShapeClass;
	extern hkClass hkpThinBoxMotionClass;
	extern hkClass hkpTransformShapeClass;
	extern hkClass hkpTriSampledHeightFieldBvTreeShapeClass;
	extern hkClass hkpTriSampledHeightFieldCollectionClass;
	extern hkClass hkpTriangleShapeClass;
	extern hkClass hkpTwistLimitConstraintAtomClass;
	extern hkClass hkpTypedBroadPhaseHandleClass;
	extern hkClass hkpTyremarkPointClass;
	extern hkClass hkpTyremarksInfoClass;
	extern hkClass hkpTyremarksWheelClass;
	extern hkClass hkpUnaryActionClass;
	extern hkClass hkpVehicleAerodynamicsClass;
	extern hkClass hkpVehicleBrakeClass;
	extern hkClass hkpVehicleDataClass;
	extern hkClass hkpVehicleDataWheelComponentParamsClass;
	extern hkClass hkpVehicleDefaultAerodynamicsClass;
	extern hkClass hkpVehicleDefaultAnalogDriverInputClass;
	extern hkClass hkpVehicleDefaultBrakeClass;
	extern hkClass hkpVehicleDefaultBrakeWheelBrakingPropertiesClass;
	extern hkClass hkpVehicleDefaultEngineClass;
	extern hkClass hkpVehicleDefaultSteeringClass;
	extern hkClass hkpVehicleDefaultSuspensionClass;
	extern hkClass hkpVehicleDefaultSuspensionWheelSpringSuspensionParametersClass;
	extern hkClass hkpVehicleDefaultTransmissionClass;
	extern hkClass hkpVehicleDefaultVelocityDamperClass;
	extern hkClass hkpVehicleDriverInputAnalogStatusClass;
	extern hkClass hkpVehicleDriverInputClass;
	extern hkClass hkpVehicleDriverInputStatusClass;
	extern hkClass hkpVehicleEngineClass;
	extern hkClass hkpVehicleFrictionDescriptionAxisDescriptionClass;
	extern hkClass hkpVehicleFrictionDescriptionClass;
	extern hkClass hkpVehicleFrictionStatusAxisStatusClass;
	extern hkClass hkpVehicleFrictionStatusClass;
	extern hkClass hkpVehicleInstanceClass;
	extern hkClass hkpVehicleInstanceWheelInfoClass;
	extern hkClass hkpVehicleRaycastWheelCollideClass;
	extern hkClass hkpVehicleSteeringClass;
	extern hkClass hkpVehicleSuspensionClass;
	extern hkClass hkpVehicleSuspensionSuspensionWheelParametersClass;
	extern hkClass hkpVehicleTransmissionClass;
	extern hkClass hkpVehicleVelocityDamperClass;
	extern hkClass hkpVehicleWheelCollideClass;
	extern hkClass hkpVelocityConstraintMotorClass;
	extern hkClass hkpViscousSurfaceModifierConstraintAtomClass;
	extern hkClass hkpWeldingUtilityClass;
	extern hkClass hkpWheelConstraintDataAtomsClass;
	extern hkClass hkpWheelConstraintDataClass;
	extern hkClass hkpWorldCinfoClass;
	extern hkClass hkpWorldObjectClass;
	extern hkClass hkxAnimatedFloatClass;
	extern hkClass hkxAnimatedMatrixClass;
	extern hkClass hkxAnimatedQuaternionClass;
	extern hkClass hkxAnimatedVectorClass;
	extern hkClass hkxAttributeClass;
	extern hkClass hkxAttributeGroupClass;
	extern hkClass hkxAttributeHolderClass;
	extern hkClass hkxCameraClass;
	extern hkClass hkxEdgeSelectionChannelClass;
	extern hkClass hkxEnvironmentClass;
	extern hkClass hkxEnvironmentVariableClass;
	extern hkClass hkxIndexBufferClass;
	extern hkClass hkxLightClass;
	extern hkClass hkxMaterialClass;
	extern hkClass hkxMaterialEffectClass;
	extern hkClass hkxMaterialShaderClass;
	extern hkClass hkxMaterialShaderSetClass;
	extern hkClass hkxMaterialTextureStageClass;
	extern hkClass hkxMeshClass;
	extern hkClass hkxMeshSectionClass;
	extern hkClass hkxMeshUserChannelInfoClass;
	extern hkClass hkxNodeAnnotationDataClass;
	extern hkClass hkxNodeClass;
	extern hkClass hkxNodeSelectionSetClass;
	extern hkClass hkxSceneClass;
	extern hkClass hkxSkinBindingClass;
	extern hkClass hkxSparselyAnimatedBoolClass;
	extern hkClass hkxSparselyAnimatedEnumClass;
	extern hkClass hkxSparselyAnimatedIntClass;
	extern hkClass hkxSparselyAnimatedStringClass;
	extern hkClass hkxSparselyAnimatedStringStringTypeClass;
	extern hkClass hkxTextureFileClass;
	extern hkClass hkxTextureInplaceClass;
	extern hkClass hkxTriangleSelectionChannelClass;
	extern hkClass hkxVertexBufferClass;
	extern hkClass hkxVertexDescriptionClass;
	extern hkClass hkxVertexDescriptionElementDeclClass;
	extern hkClass hkxVertexFloatDataChannelClass;
	extern hkClass hkxVertexIntDataChannelClass;
	extern hkClass hkxVertexP4N4C1T10Class;
	extern hkClass hkxVertexP4N4C1T2Class;
	extern hkClass hkxVertexP4N4C1T6Class;
	extern hkClass hkxVertexP4N4T4B4C1T10Class;
	extern hkClass hkxVertexP4N4T4B4C1T2Class;
	extern hkClass hkxVertexP4N4T4B4C1T6Class;
	extern hkClass hkxVertexP4N4T4B4T4Class;
	extern hkClass hkxVertexP4N4T4B4W4I4C1Q2Class;
	extern hkClass hkxVertexP4N4T4B4W4I4C1T12Class;
	extern hkClass hkxVertexP4N4T4B4W4I4C1T4Class;
	extern hkClass hkxVertexP4N4T4B4W4I4C1T8Class;
	extern hkClass hkxVertexP4N4T4B4W4I4Q4Class;
	extern hkClass hkxVertexP4N4T4B4W4I4T6Class;
	extern hkClass hkxVertexP4N4T4Class;
	extern hkClass hkxVertexP4N4W4I4C1Q2Class;
	extern hkClass hkxVertexP4N4W4I4C1T12Class;
	extern hkClass hkxVertexP4N4W4I4C1T4Class;
	extern hkClass hkxVertexP4N4W4I4C1T8Class;
	extern hkClass hkxVertexP4N4W4I4T2Class;
	extern hkClass hkxVertexP4N4W4I4T6Class;
	extern hkClass hkxVertexSelectionChannelClass;
	extern hkClass hkxVertexVectorDataChannelClass;

	static const hkInternalClassEnumItem ManifesthclBufferTypeEnumItems[] =
	{
		{0, "HCL_BUFFER_TYPE_INVALID"},
		{1, "HCL_BUFFER_TYPE_CLOTH_CURRENT_POSITIONS"},
		{2, "HCL_BUFFER_TYPE_CLOTH_PREVIOUS_POSITIONS"},
		{3, "HCL_BUFFER_TYPE_CLOTHDATA_POSE"},
		{4, "HCL_BUFFER_TYPE_DISPLAY"},
		{5, "HCL_BUFFER_TYPE_STATIC_DISPLAY"},
		{6, "HCL_BUFFER_TYPE_SCRATCH_POSITIONS_BUFFER"},
		{100, "HCL_BUFFER_TYPE_USER_0"},
		{101, "HCL_BUFFER_TYPE_USER_1"},
	};
	static const hkInternalClassEnumItem ManifesthclShapeTypeEnumItems[] =
	{
		{0, "HCL_SHAPE_TYPE_SPHERE"},
		{1, "HCL_SHAPE_TYPE_PLANE"},
		{2, "HCL_SHAPE_TYPE_CAPSULE"},
		{3, "HCL_SHAPE_TYPE_CYLINDER"},
		{4, "HCL_SHAPE_TYPE_CONVEX_HEIGHTFIELD"},
		{5, "HCL_SHAPE_TYPE_CONNECTED_MESH"},
		{6, "HCL_SHAPE_TYPE_PLANAR_HEIGHTFIELD"},
		{7, "HCL_SHAPE_TYPE_DEFORMING_MESH"},
	};
	static const hkInternalClassEnumItem ManifesthclTransformSetTypeEnumItems[] =
	{
		{0, "HCL_TRANSFORMSET_TYPE_INVALID"},
		{1, "HCL_TRANSFORMSET_TYPE_ANIMATION"},
		{2, "HCL_TRANSFORMSET_TYPE_COLLIDABLES"},
		{100, "HCL_TRANSFORMSET_TYPE_USER_0"},
		{101, "HCL_TRANSFORMSET_TYPE_USER_1"},
	};
	static const hkInternalClassEnumItem ManifesthkpUpdateCollisionFilterOnWorldModeEnumItems[] =
	{
		{0, "HK_UPDATE_FILTER_ON_WORLD_FULL_CHECK"},
		{1, "HK_UPDATE_FILTER_ON_WORLD_DISABLE_ENTITY_ENTITY_COLLISIONS_ONLY"},
	};
	static const hkInternalClassEnumItem ManifesthkpUpdateCollisionFilterOnEntityModeEnumItems[] =
	{
		{0, "HK_UPDATE_FILTER_ON_ENTITY_FULL_CHECK"},
		{1, "HK_UPDATE_FILTER_ON_ENTITY_DISABLE_ENTITY_ENTITY_COLLISIONS_ONLY"},
	};
	static const hkInternalClassEnumItem ManifesthkpEntityActivationEnumItems[] =
	{
		{0, "HK_ENTITY_ACTIVATION_DO_NOT_ACTIVATE"},
		{1, "HK_ENTITY_ACTIVATION_DO_ACTIVATE"},
	};
	static const hkInternalClassEnumItem ManifesthkpUpdateCollectionFilterModeEnumItems[] =
	{
		{0, "HK_UPDATE_COLLECTION_FILTER_IGNORE_SHAPE_COLLECTIONS"},
		{1, "HK_UPDATE_COLLECTION_FILTER_PROCESS_SHAPE_COLLECTIONS"},
	};
	static const hkInternalClassEnumItem ManifesthkpStepResultEnumItems[] =
	{
		{0, "HK_STEP_RESULT_SUCCESS"},
		{1, "HK_STEP_RESULT_MEMORY_FAILURE_BEFORE_INTEGRATION"},
		{2, "HK_STEP_RESULT_MEMORY_FAILURE_DURING_COLLIDE"},
		{3, "HK_STEP_RESULT_MEMORY_FAILURE_DURING_TOI_SOLVE"},
	};
	static const hkInternalClassEnumItem ManifestResultEnumItems[] =
	{
		{0, "POSTPONED"},
		{1, "DONE"},
	};
	static const hkInternalClassEnum ManifestEnums[] = {
		{"hclBufferType", ManifesthclBufferTypeEnumItems, 9, HK_NULL, 0 },
		{"hclShapeType", ManifesthclShapeTypeEnumItems, 8, HK_NULL, 0 },
		{"hclTransformSetType", ManifesthclTransformSetTypeEnumItems, 5, HK_NULL, 0 },
		{"hkpUpdateCollisionFilterOnWorldMode", ManifesthkpUpdateCollisionFilterOnWorldModeEnumItems, 2, HK_NULL, 0 },
		{"hkpUpdateCollisionFilterOnEntityMode", ManifesthkpUpdateCollisionFilterOnEntityModeEnumItems, 2, HK_NULL, 0 },
		{"hkpEntityActivation", ManifesthkpEntityActivationEnumItems, 2, HK_NULL, 0 },
		{"hkpUpdateCollectionFilterMode", ManifesthkpUpdateCollectionFilterModeEnumItems, 2, HK_NULL, 0 },
		{"hkpStepResult", ManifesthkpStepResultEnumItems, 4, HK_NULL, 0 },
		{"Result", ManifestResultEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hclBufferTypeEnum = reinterpret_cast<const hkClassEnum*>(&ManifestEnums[0]);
	const hkClassEnum* hclShapeTypeEnum = reinterpret_cast<const hkClassEnum*>(&ManifestEnums[1]);
	const hkClassEnum* hclTransformSetTypeEnum = reinterpret_cast<const hkClassEnum*>(&ManifestEnums[2]);
	const hkClassEnum* hkpUpdateCollisionFilterOnWorldModeEnum = reinterpret_cast<const hkClassEnum*>(&ManifestEnums[3]);
	const hkClassEnum* hkpUpdateCollisionFilterOnEntityModeEnum = reinterpret_cast<const hkClassEnum*>(&ManifestEnums[4]);
	const hkClassEnum* hkpEntityActivationEnum = reinterpret_cast<const hkClassEnum*>(&ManifestEnums[5]);
	const hkClassEnum* hkpUpdateCollectionFilterModeEnum = reinterpret_cast<const hkClassEnum*>(&ManifestEnums[6]);
	const hkClassEnum* hkpStepResultEnum = reinterpret_cast<const hkClassEnum*>(&ManifestEnums[7]);
	const hkClassEnum* ResultEnum = reinterpret_cast<const hkClassEnum*>(&ManifestEnums[8]);
	extern const hkClassEnum* hclConstraintSetTypeEnum;
	extern const hkClassEnum* hclOperatorTypeEnum;
	extern const hkClassEnum* hkClassMemberTypeEnum;
	extern const hkClassEnum* hkColorExtendedColorsEnum;
	extern const hkClassEnum* hkaAnimationAnimationTypeEnum;
	extern const hkClassEnum* hkbBlendCurveUtilsBlendCurveEnum;
	extern const hkClassEnum* hkbConstrainRigidBodyModifierBoneToConstrainPlacementEnum;
	extern const hkClassEnum* hkbStateMachineTransitionInfoTransitionFlagsEnum;
	extern const hkClassEnum* hkdFractureConnectivityTypeEnum;
	extern const hkClassEnum* hkdFractureRefitPhysicsTypeEnum;
	extern const hkClassEnum* hkdShapeConnectivityEnum;
	extern const hkClassEnum* hkdShapeIntegrityTypeEnum;
	extern const hkClassEnum* hkpConstraintDataConstraintTypeEnum;
	extern const hkClassEnum* hkpConstraintInstanceConstraintPriorityEnum;
	extern const hkClassEnum* hkpShapeTypeEnum;
	extern const hkClassEnum* hkpWeldingUtilityWeldingTypeEnum;
	extern const hkClassEnum* hkxAttributeHintEnum;
	extern const hkClassEnum* hkxVertexFloatDataChannelVertexFloatDimensionsEnum;
	static hkInternalClassMember hkaAnimationContainerClass_Members[] =
	{
		{ "skeletons", &hkaSkeletonClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "animations", &hkaAnimationClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "bindings", &hkaAnimationBindingClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "attachments", &hkaBoneAttachmentClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "skins", &hkaMeshBindingClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkaAnimationContainerClass(
		"hkaAnimationContainer",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaAnimationContainerClass_Members),
		HK_COUNT_OF(hkaAnimationContainerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkaAnimationAnimationTypeEnumItems[] =
	{
		{0, "HK_UNKNOWN_ANIMATION"},
		{1, "HK_INTERLEAVED_ANIMATION"},
		{2, "HK_DELTA_COMPRESSED_ANIMATION"},
		{3, "HK_WAVELET_COMPRESSED_ANIMATION"},
		{4, "HK_MIRRORED_ANIMATION"},
		{5, "HK_SPLINE_COMPRESSED_ANIMATION"},
	};
	static const hkInternalClassEnum hkaAnimationEnums[] = {
		{"AnimationType", hkaAnimationAnimationTypeEnumItems, 6, HK_NULL, 0 }
	};
	const hkClassEnum* hkaAnimationAnimationTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkaAnimationEnums[0]);
	static hkInternalClassMember hkaAnimationClass_Members[] =
	{
		{ "type", HK_NULL, hkaAnimationAnimationTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "duration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numberOfTransformTracks", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numberOfFloatTracks", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extractedMotion", &hkaAnimatedReferenceFrameClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "annotationTracks", &hkaAnnotationTrackClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkaAnimationClass(
		"hkaAnimation",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkaAnimationEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkaAnimationClass_Members),
		HK_COUNT_OF(hkaAnimationClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkaAnimationBindingBlendHintEnumItems[] =
	{
		{0, "NORMAL"},
		{1, "ADDITIVE"},
	};
	static const hkInternalClassEnum hkaAnimationBindingEnums[] = {
		{"BlendHint", hkaAnimationBindingBlendHintEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkaAnimationBindingBlendHintEnum = reinterpret_cast<const hkClassEnum*>(&hkaAnimationBindingEnums[0]);
	static hkInternalClassMember hkaAnimationBindingClass_Members[] =
	{
		{ "originalSkeletonName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "animation", &hkaAnimationClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "transformTrackToBoneIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "floatTrackToFloatSlotIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "blendHint", HK_NULL, hkaAnimationBindingBlendHintEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkaAnimationBindingClass(
		"hkaAnimationBinding",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkaAnimationBindingEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkaAnimationBindingClass_Members),
		HK_COUNT_OF(hkaAnimationBindingClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaAnnotationTrack_AnnotationClass_Members[] =
	{
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "text", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaAnnotationTrackAnnotationClass(
		"hkaAnnotationTrackAnnotation",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaAnnotationTrack_AnnotationClass_Members),
		HK_COUNT_OF(hkaAnnotationTrack_AnnotationClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaAnnotationTrackClass_Members[] =
	{
		{ "trackName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "annotations", &hkaAnnotationTrackAnnotationClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkaAnnotationTrackClass(
		"hkaAnnotationTrack",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaAnnotationTrackClass_Members),
		HK_COUNT_OF(hkaAnnotationTrackClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaDeltaCompressedAnimation_QuantizationFormatClass_Members[] =
	{
		{ "maxBitWidth", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "preserved", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numD", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "offsetIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "scaleIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bitWidthIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaDeltaCompressedAnimationQuantizationFormatClass(
		"hkaDeltaCompressedAnimationQuantizationFormat",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaDeltaCompressedAnimation_QuantizationFormatClass_Members),
		HK_COUNT_OF(hkaDeltaCompressedAnimation_QuantizationFormatClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaDeltaCompressedAnimationClass_Members[] =
	{
		{ "numberOfPoses", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blockSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "qFormat", &hkaDeltaCompressedAnimationQuantizationFormatClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "quantizedDataIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "quantizedDataSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "staticMaskIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "staticMaskSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "staticDOFsIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "staticDOFsSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numStaticTransformDOFs", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numDynamicTransformDOFs", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "totalBlockSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lastBlockSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dataBuffer", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkaDeltaCompressedAnimationClass(
		"hkaDeltaCompressedAnimation",
		&hkaAnimationClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaDeltaCompressedAnimationClass_Members),
		HK_COUNT_OF(hkaDeltaCompressedAnimationClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaInterleavedUncompressedAnimationClass_Members[] =
	{
		{ "transforms", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_QSTRANSFORM, 0, 0, 0, HK_NULL },
		{ "floats", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL }
	};
	hkClass hkaInterleavedUncompressedAnimationClass(
		"hkaInterleavedUncompressedAnimation",
		&hkaAnimationClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaInterleavedUncompressedAnimationClass_Members),
		HK_COUNT_OF(hkaInterleavedUncompressedAnimationClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkaSplineCompressedAnimationTrackCompressionParamsRotationQuantizationEnumItems[] =
	{
		{0, "POLAR32"},
		{1, "THREECOMP40"},
		{2, "THREECOMP48"},
		{3, "THREECOMP24"},
		{4, "STRAIGHT16"},
		{5, "UNCOMPRESSED"},
	};
	static const hkInternalClassEnumItem hkaSplineCompressedAnimationTrackCompressionParamsScalarQuantizationEnumItems[] =
	{
		{0, "BITS8"},
		{1, "BITS16"},
	};
	static const hkInternalClassEnum hkaSplineCompressedAnimationTrackCompressionParamsEnums[] = {
		{"RotationQuantization", hkaSplineCompressedAnimationTrackCompressionParamsRotationQuantizationEnumItems, 6, HK_NULL, 0 },
		{"ScalarQuantization", hkaSplineCompressedAnimationTrackCompressionParamsScalarQuantizationEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkaSplineCompressedAnimationTrackCompressionParamsRotationQuantizationEnum = reinterpret_cast<const hkClassEnum*>(&hkaSplineCompressedAnimationTrackCompressionParamsEnums[0]);
	const hkClassEnum* hkaSplineCompressedAnimationTrackCompressionParamsScalarQuantizationEnum = reinterpret_cast<const hkClassEnum*>(&hkaSplineCompressedAnimationTrackCompressionParamsEnums[1]);
	static hkInternalClassMember hkaSplineCompressedAnimation_TrackCompressionParamsClass_Members[] =
	{
		{ "rotationTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "translationTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "scaleTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "floatingTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotationDegree", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "translationDegree", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "scaleDegree", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "floatingDegree", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotationQuantizationType", HK_NULL, hkaSplineCompressedAnimationTrackCompressionParamsRotationQuantizationEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "translationQuantizationType", HK_NULL, hkaSplineCompressedAnimationTrackCompressionParamsScalarQuantizationEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "scaleQuantizationType", HK_NULL, hkaSplineCompressedAnimationTrackCompressionParamsScalarQuantizationEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "floatQuantizationType", HK_NULL, hkaSplineCompressedAnimationTrackCompressionParamsScalarQuantizationEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkaSplineCompressedAnimationTrackCompressionParamsClass(
		"hkaSplineCompressedAnimationTrackCompressionParams",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkaSplineCompressedAnimationTrackCompressionParamsEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkaSplineCompressedAnimation_TrackCompressionParamsClass_Members),
		HK_COUNT_OF(hkaSplineCompressedAnimation_TrackCompressionParamsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaSplineCompressedAnimation_AnimationCompressionParamsClass_Members[] =
	{
		{ "maxFramesPerBlock", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "enableSampleSingleTracks", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaSplineCompressedAnimationAnimationCompressionParamsClass(
		"hkaSplineCompressedAnimationAnimationCompressionParams",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaSplineCompressedAnimation_AnimationCompressionParamsClass_Members),
		HK_COUNT_OF(hkaSplineCompressedAnimation_AnimationCompressionParamsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaSplineCompressedAnimationClass_Members[] =
	{
		{ "numFrames", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numBlocks", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxFramesPerBlock", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maskAndQuantizationSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blockDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blockInverseDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "frameDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blockOffsets", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "floatBlockOffsets", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "transformOffsets", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "floatOffsets", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "endian", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaSplineCompressedAnimationClass(
		"hkaSplineCompressedAnimation",
		&hkaAnimationClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaSplineCompressedAnimationClass_Members),
		HK_COUNT_OF(hkaSplineCompressedAnimationClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaAnimationPreviewColorClass_Members[] =
	{
		{ "color", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaAnimationPreviewColorClass(
		"hkaAnimationPreviewColor",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaAnimationPreviewColorClass_Members),
		HK_COUNT_OF(hkaAnimationPreviewColorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaAnimationPreviewColorContainerClass_Members[] =
	{
		{ "previewColor", &hkaAnimationPreviewColorClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkaAnimationPreviewColorContainerClass(
		"hkaAnimationPreviewColorContainer",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaAnimationPreviewColorContainerClass_Members),
		HK_COUNT_OF(hkaAnimationPreviewColorContainerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaWaveletCompressedAnimation_CompressionParamsClass_Members[] =
	{
		{ "quantizationBits", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blockSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "preserve", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "truncProp", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useOldStyleTruncation", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "absolutePositionTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "relativePositionTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotationTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "scaleTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "absoluteFloatTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaWaveletCompressedAnimationCompressionParamsClass(
		"hkaWaveletCompressedAnimationCompressionParams",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaWaveletCompressedAnimation_CompressionParamsClass_Members),
		HK_COUNT_OF(hkaWaveletCompressedAnimation_CompressionParamsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaWaveletCompressedAnimation_QuantizationFormatClass_Members[] =
	{
		{ "maxBitWidth", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "preserved", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numD", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "offsetIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "scaleIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bitWidthIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaWaveletCompressedAnimationQuantizationFormatClass(
		"hkaWaveletCompressedAnimationQuantizationFormat",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaWaveletCompressedAnimation_QuantizationFormatClass_Members),
		HK_COUNT_OF(hkaWaveletCompressedAnimation_QuantizationFormatClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaWaveletCompressedAnimationClass_Members[] =
	{
		{ "numberOfPoses", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blockSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "qFormat", &hkaWaveletCompressedAnimationQuantizationFormatClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "staticMaskIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "staticDOFsIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numStaticTransformDOFs", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numDynamicTransformDOFs", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blockIndexIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blockIndexSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "quantizedDataIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "quantizedDataSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dataBuffer", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkaWaveletCompressedAnimationClass(
		"hkaWaveletCompressedAnimation",
		&hkaAnimationClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaWaveletCompressedAnimationClass_Members),
		HK_COUNT_OF(hkaWaveletCompressedAnimationClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaMeshBinding_MappingClass_Members[] =
	{
		{ "mapping", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL }
	};
	hkClass hkaMeshBindingMappingClass(
		"hkaMeshBindingMapping",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaMeshBinding_MappingClass_Members),
		HK_COUNT_OF(hkaMeshBinding_MappingClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaMeshBindingClass_Members[] =
	{
		{ "mesh", &hkxMeshClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "originalSkeletonName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "skeleton", &hkaSkeletonClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "mappings", &hkaMeshBindingMappingClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "boneFromSkinMeshTransforms", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_TRANSFORM, 0, 0, 0, HK_NULL }
	};
	hkClass hkaMeshBindingClass(
		"hkaMeshBinding",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaMeshBindingClass_Members),
		HK_COUNT_OF(hkaMeshBindingClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkaSkeletonMapperConstraintSourceEnumItems[] =
	{
		{0, "NO_CONSTRAINTS"},
		{1, "REFERENCE_POSE"},
		{2, "CURRENT_POSE"},
	};
	static const hkInternalClassEnum hkaSkeletonMapperEnums[] = {
		{"ConstraintSource", hkaSkeletonMapperConstraintSourceEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkaSkeletonMapperConstraintSourceEnum = reinterpret_cast<const hkClassEnum*>(&hkaSkeletonMapperEnums[0]);
	static hkInternalClassMember hkaSkeletonMapperClass_Members[] =
	{
		{ "mapping", &hkaSkeletonMapperDataClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaSkeletonMapperClass(
		"hkaSkeletonMapper",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkaSkeletonMapperEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkaSkeletonMapperClass_Members),
		HK_COUNT_OF(hkaSkeletonMapperClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaSkeletonMapperData_SimpleMappingClass_Members[] =
	{
		{ "boneA", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boneB", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aFromBTransform", HK_NULL, HK_NULL, hkClassMember::TYPE_QSTRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaSkeletonMapperDataSimpleMappingClass(
		"hkaSkeletonMapperDataSimpleMapping",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaSkeletonMapperData_SimpleMappingClass_Members),
		HK_COUNT_OF(hkaSkeletonMapperData_SimpleMappingClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaSkeletonMapperData_ChainMappingClass_Members[] =
	{
		{ "startBoneA", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "endBoneA", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "startBoneB", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "endBoneB", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "startAFromBTransform", HK_NULL, HK_NULL, hkClassMember::TYPE_QSTRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "endAFromBTransform", HK_NULL, HK_NULL, hkClassMember::TYPE_QSTRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaSkeletonMapperDataChainMappingClass(
		"hkaSkeletonMapperDataChainMapping",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaSkeletonMapperData_ChainMappingClass_Members),
		HK_COUNT_OF(hkaSkeletonMapperData_ChainMappingClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaSkeletonMapperDataClass_Members[] =
	{
		{ "skeletonA", &hkaSkeletonClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "skeletonB", &hkaSkeletonClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "simpleMappings", &hkaSkeletonMapperDataSimpleMappingClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "chainMappings", &hkaSkeletonMapperDataChainMappingClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "unmappedBones", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "keepUnmappedLocal", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaSkeletonMapperDataClass(
		"hkaSkeletonMapperData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaSkeletonMapperDataClass_Members),
		HK_COUNT_OF(hkaSkeletonMapperDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkaAnimatedReferenceFrameClass(
		"hkaAnimatedReferenceFrame",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaDefaultAnimatedReferenceFrameClass_Members[] =
	{
		{ "up", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "forward", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "duration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "referenceFrameSamples", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL }
	};
	hkClass hkaDefaultAnimatedReferenceFrameClass(
		"hkaDefaultAnimatedReferenceFrame",
		&hkaAnimatedReferenceFrameClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaDefaultAnimatedReferenceFrameClass_Members),
		HK_COUNT_OF(hkaDefaultAnimatedReferenceFrameClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaFootstepAnalysisInfoClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CHAR, 0, 0, 0, HK_NULL },
		{ "nameStrike", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CHAR, 0, 0, 0, HK_NULL },
		{ "nameLift", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CHAR, 0, 0, 0, HK_NULL },
		{ "nameLock", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CHAR, 0, 0, 0, HK_NULL },
		{ "nameUnlock", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CHAR, 0, 0, 0, HK_NULL },
		{ "minPos", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "maxPos", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "minVel", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "maxVel", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "allBonesDown", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "anyBonesDown", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "posTol", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "velTol", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "duration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaFootstepAnalysisInfoClass(
		"hkaFootstepAnalysisInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaFootstepAnalysisInfoClass_Members),
		HK_COUNT_OF(hkaFootstepAnalysisInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaFootstepAnalysisInfoContainerClass_Members[] =
	{
		{ "previewInfo", &hkaFootstepAnalysisInfoClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkaFootstepAnalysisInfoContainerClass(
		"hkaFootstepAnalysisInfoContainer",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaFootstepAnalysisInfoContainerClass_Members),
		HK_COUNT_OF(hkaFootstepAnalysisInfoContainerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaBoneClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lockTranslation", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaBoneClass(
		"hkaBone",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaBoneClass_Members),
		HK_COUNT_OF(hkaBoneClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaBoneAttachmentClass_Members[] =
	{
		{ "originalSkeletonName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boneFromAttachment", HK_NULL, HK_NULL, hkClassMember::TYPE_MATRIX4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "attachment", HK_NULL, HK_NULL, hkClassMember::TYPE_VARIANT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaBoneAttachmentClass(
		"hkaBoneAttachment",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaBoneAttachmentClass_Members),
		HK_COUNT_OF(hkaBoneAttachmentClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaSkeleton_LocalFrameOnBoneClass_Members[] =
	{
		{ "localFrame", &hkLocalFrameClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "boneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkaSkeletonLocalFrameOnBoneClass(
		"hkaSkeletonLocalFrameOnBone",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaSkeleton_LocalFrameOnBoneClass_Members),
		HK_COUNT_OF(hkaSkeleton_LocalFrameOnBoneClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaSkeletonClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "parentIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "bones", &hkaBoneClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "referencePose", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_QSTRANSFORM, 0, 0, 0, HK_NULL },
		{ "floatSlots", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "localFrames", &hkaSkeletonLocalFrameOnBoneClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkaSkeletonClass(
		"hkaSkeleton",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaSkeletonClass_Members),
		HK_COUNT_OF(hkaSkeletonClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaKeyFrameHierarchyUtility_ControlDataClass_Members[] =
	{
		{ "hierarchyGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "velocityDamping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "accelerationGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "velocityGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "positionGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "positionMaxLinearVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "positionMaxAngularVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "snapGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "snapMaxLinearVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "snapMaxAngularVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "snapMaxLinearDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "snapMaxAngularDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkaKeyFrameHierarchyUtilityControlData_DefaultStruct
		{
			int s_defaultOffsets[12];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_hierarchyGain;
			hkReal m_accelerationGain;
			hkReal m_velocityGain;
			hkReal m_positionGain;
			hkReal m_positionMaxLinearVelocity;
			hkReal m_positionMaxAngularVelocity;
			hkReal m_snapGain;
			hkReal m_snapMaxLinearVelocity;
			hkReal m_snapMaxAngularVelocity;
			hkReal m_snapMaxLinearDistance;
			hkReal m_snapMaxAngularDistance;
		};
		const hkaKeyFrameHierarchyUtilityControlData_DefaultStruct hkaKeyFrameHierarchyUtilityControlData_Default =
		{
			{HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_hierarchyGain),-1,HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_accelerationGain),HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_velocityGain),HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_positionGain),HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_positionMaxLinearVelocity),HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_positionMaxAngularVelocity),HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_snapGain),HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_snapMaxLinearVelocity),HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_snapMaxAngularVelocity),HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_snapMaxLinearDistance),HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_snapMaxAngularDistance)},
			0.17f,1.0f,0.6f,0.05f,1.4f,1.8f,0.1f,0.3f,0.3f,0.03f,0.1f
		};
	}
	hkClass hkaKeyFrameHierarchyUtilityControlDataClass(
		"hkaKeyFrameHierarchyUtilityControlData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaKeyFrameHierarchyUtility_ControlDataClass_Members),
		HK_COUNT_OF(hkaKeyFrameHierarchyUtility_ControlDataClass_Members),
		&hkaKeyFrameHierarchyUtilityControlData_Default,
		HK_NULL,
		0,
		0
		);
	hkClass hkaKeyFrameHierarchyUtilityClass(
		"hkaKeyFrameHierarchyUtility",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkaRagdollInstanceClass_Members[] =
	{
		{ "rigidBodies", &hkpRigidBodyClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "constraints", &hkpConstraintInstanceClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "boneToRigidBodyMap", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "skeleton", &hkaSkeletonClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkaRagdollInstanceClass(
		"hkaRagdollInstance",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkaRagdollInstanceClass_Members),
		HK_COUNT_OF(hkaRagdollInstanceClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbAttachmentSetupAttachmentTypeEnumItems[] =
	{
		{0, "ATTACHMENT_TYPE_KEYFRAME_RIGID_BODY"},
		{1, "ATTACHMENT_TYPE_BALL_SOCKET_CONSTRAINT"},
		{2, "ATTACHMENT_TYPE_RAGDOLL_CONSTRAINT"},
		{3, "ATTACHMENT_TYPE_SET_WORLD_FROM_MODEL"},
	};
	static const hkInternalClassEnum hkbAttachmentSetupEnums[] = {
		{"AttachmentType", hkbAttachmentSetupAttachmentTypeEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkbAttachmentSetupAttachmentTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkbAttachmentSetupEnums[0]);
	static hkInternalClassMember hkbAttachmentSetupClass_Members[] =
	{
		{ "blendInTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "moveAttacherFraction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "gain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "attachmentType", HK_NULL, hkbAttachmentSetupAttachmentTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbAttachmentSetup_DefaultStruct
		{
			int s_defaultOffsets[4];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_gain;
		};
		const hkbAttachmentSetup_DefaultStruct hkbAttachmentSetup_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkbAttachmentSetup_DefaultStruct,m_gain),-1},
			0.3f
		};
	}
	hkClass hkbAttachmentSetupClass(
		"hkbAttachmentSetup",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbAttachmentSetupEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbAttachmentSetupClass_Members),
		HK_COUNT_OF(hkbAttachmentSetupClass_Members),
		&hkbAttachmentSetup_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbBehaviorGraphVariableModeEnumItems[] =
	{
		{0, "VARIABLE_MODE_DISCARD_WHEN_INACTIVE"},
		{1, "VARIABLE_MODE_MAINTAIN_MEMORY_WHEN_INACTIVE"},
		{2, "VARIABLE_MODE_MAINTAIN_VALUES_WHEN_INACTIVE"},
	};
	static const hkInternalClassEnum hkbBehaviorGraphEnums[] = {
		{"VariableMode", hkbBehaviorGraphVariableModeEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkbBehaviorGraphVariableModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbBehaviorGraphEnums[0]);
	static hkInternalClassMember hkbBehaviorGraphClass_Members[] =
	{
		{ "variableMode", HK_NULL, hkbBehaviorGraphVariableModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "rootGenerator", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "data", &hkbBehaviorGraphDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "rootGeneratorClone", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isActive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "activeNodes", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "activeNodeToIndexMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "activeNodesChildrenIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "eventIdMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "attributeIdMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "variableIdMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "variableValues", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "quadVariableValues", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "mirroredExternalIdMap", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbBehaviorGraphClass(
		"hkbBehaviorGraph",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbBehaviorGraphEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbBehaviorGraphClass_Members),
		HK_COUNT_OF(hkbBehaviorGraphClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbBehaviorGraphDataClass_Members[] =
	{
		{ "attributeDefaults", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "variableInfos", &hkbVariableInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "quadVariableInitialValues", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL },
		{ "eventInfos", &hkbEventInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "stringData", &hkbBehaviorGraphStringDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbBehaviorGraphDataClass(
		"hkbBehaviorGraphData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbBehaviorGraphDataClass_Members),
		HK_COUNT_OF(hkbBehaviorGraphDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbBehaviorGraphStringDataClass_Members[] =
	{
		{ "eventNames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "attributeNames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "variableNames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL }
	};
	hkClass hkbBehaviorGraphStringDataClass(
		"hkbBehaviorGraphStringData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbBehaviorGraphStringDataClass_Members),
		HK_COUNT_OF(hkbBehaviorGraphStringDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkbCharacterClass(
		"hkbCharacter",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbCharacterBoneInfoWhichLimbEnumItems[] =
	{
		{0, "FIRST_LIMB"},
		{0, "LIMB_LEFT"},
		{1, "LIMB_RIGHT"},
		{2, "NUM_LIMBS"},
	};
	static const hkInternalClassEnum hkbCharacterBoneInfoEnums[] = {
		{"WhichLimb", hkbCharacterBoneInfoWhichLimbEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkbCharacterBoneInfoWhichLimbEnum = reinterpret_cast<const hkClassEnum*>(&hkbCharacterBoneInfoEnums[0]);
	static hkInternalClassMember hkbCharacterBoneInfoClass_Members[] =
	{
		{ "clavicleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "shoulderIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "elbowIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "wristIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "hipIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "kneeIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "ankleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "spineIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "pelvisIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "neckIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "headIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "poseMatchingRootBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "poseMatchingOtherBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "poseMatchingAnotherBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbCharacterBoneInfo_DefaultStruct
		{
			int s_defaultOffsets[14];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt16 m_poseMatchingRootBoneIndex;
			hkInt16 m_poseMatchingOtherBoneIndex;
			hkInt16 m_poseMatchingAnotherBoneIndex;
		};
		const hkbCharacterBoneInfo_DefaultStruct hkbCharacterBoneInfo_Default =
		{
			{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkbCharacterBoneInfo_DefaultStruct,m_poseMatchingRootBoneIndex),HK_OFFSET_OF(hkbCharacterBoneInfo_DefaultStruct,m_poseMatchingOtherBoneIndex),HK_OFFSET_OF(hkbCharacterBoneInfo_DefaultStruct,m_poseMatchingAnotherBoneIndex)},
			-1,-1,-1
		};
	}
	hkClass hkbCharacterBoneInfoClass(
		"hkbCharacterBoneInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbCharacterBoneInfoEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbCharacterBoneInfoClass_Members),
		HK_COUNT_OF(hkbCharacterBoneInfoClass_Members),
		&hkbCharacterBoneInfo_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbCharacterDataClass_Members[] =
	{
		{ "animationBoneInfo", &hkbCharacterBoneInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollBoneInfo", &hkbCharacterBoneInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "modelUpMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "modelForwardMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "modelRightMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stringData", &hkbCharacterStringDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "mirroredSkeletonInfo", &hkbMirroredSkeletonInfoClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbCharacterDataClass(
		"hkbCharacterData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbCharacterDataClass_Members),
		HK_COUNT_OF(hkbCharacterDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbCharacterSetupClass_Members[] =
	{
		{ "animationSkeleton", &hkaSkeletonClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "ragdollSkeleton", &hkaSkeletonClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "ragdollToAnimationSkeletonMapper", &hkaSkeletonMapperClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "animationToRagdollSkeletonMapper", &hkaSkeletonMapperClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "data", &hkbCharacterDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "mirroredSkeleton", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbCharacterSetupClass(
		"hkbCharacterSetup",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbCharacterSetupClass_Members),
		HK_COUNT_OF(hkbCharacterSetupClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbCharacterStringDataClass_Members[] =
	{
		{ "deformableSkinNames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "rigidSkinNames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "animationNames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "animationFilenames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rigName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbCharacterStringDataClass(
		"hkbCharacterStringData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbCharacterStringDataClass_Members),
		HK_COUNT_OF(hkbCharacterStringDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbMirroredSkeletonInfoClass_Members[] =
	{
		{ "mirrorAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bonePairMap", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbMirroredSkeletonInfo_DefaultStruct
		{
			int s_defaultOffsets[2];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_mirrorAxis;
		};
		const hkbMirroredSkeletonInfo_DefaultStruct hkbMirroredSkeletonInfo_Default =
		{
			{HK_OFFSET_OF(hkbMirroredSkeletonInfo_DefaultStruct,m_mirrorAxis),-1},
			{1,0,0,0}
		};
	}
	hkClass hkbMirroredSkeletonInfoClass(
		"hkbMirroredSkeletonInfo",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbMirroredSkeletonInfoClass_Members),
		HK_COUNT_OF(hkbMirroredSkeletonInfoClass_Members),
		&hkbMirroredSkeletonInfo_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbContextClass_Members[] =
	{
		{ "character", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "rootBehavior", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "behavior", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "nodeToIndexMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "generatorOutputListener", &hkbGeneratorOutputListenerClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "projectData", &hkbProjectDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "world", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "attachmentManager", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "animationCache", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbContextClass(
		"hkbContext",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbContextClass_Members),
		HK_COUNT_OF(hkbContextClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbEventSystemEventIdsEnumItems[] =
	{
		{-1, "EVENT_ID_NULL"},
	};
	static const hkInternalClassEnum hkbEventEnums[] = {
		{"SystemEventIds", hkbEventSystemEventIdsEnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hkbEventSystemEventIdsEnum = reinterpret_cast<const hkClassEnum*>(&hkbEventEnums[0]);
	static hkInternalClassMember hkbEventClass_Members[] =
	{
		{ "id", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "payload", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbEvent_DefaultStruct
		{
			int s_defaultOffsets[2];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_id;
		};
		const hkbEvent_DefaultStruct hkbEvent_Default =
		{
			{HK_OFFSET_OF(hkbEvent_DefaultStruct,m_id),-1},
			-1
		};
	}
	hkClass hkbEventClass(
		"hkbEvent",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbEventEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbEventClass_Members),
		HK_COUNT_OF(hkbEventClass_Members),
		&hkbEvent_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbEventInfoFlagsEnumItems[] =
	{
		{1/*0x1*/, "FLAG_SILENT"},
		{2/*0x2*/, "FLAG_SYNC_POINT"},
	};
	static const hkInternalClassEnum hkbEventInfoEnums[] = {
		{"Flags", hkbEventInfoFlagsEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbEventInfoFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkbEventInfoEnums[0]);
	static hkInternalClassMember hkbEventInfoClass_Members[] =
	{
		{ "flags", HK_NULL, hkbEventInfoFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL }
	};
	hkClass hkbEventInfoClass(
		"hkbEventInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbEventInfoEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbEventInfoClass_Members),
		HK_COUNT_OF(hkbEventInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkbGeneratorClass(
		"hkbGenerator",
		&hkbNodeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbGeneratorOutputStandardTracksEnumItems[] =
	{
		{0, "TRACK_WORLD_FROM_MODEL"},
		{1, "TRACK_POSE"},
		{2, "TRACK_FLOAT_SLOTS"},
		{3, "TRACK_RIGID_BODY_RAGDOLL_CONTROLS"},
		{4, "TRACK_POWERED_RAGDOLL_CONTROLS"},
		{5, "TRACK_POWERED_RAGDOLL_BONE_WEIGHTS"},
		{6, "TRACK_KEYFRAMED_RAGDOLL_BONES"},
		{7, "TRACK_ATTRIBUTES"},
		{8, "TRACK_FOOT_IK_CONTROLS"},
		{9, "TRACK_BONE_FORCES"},
		{10, "TRACK_HAND_IK_CONTROLS_0"},
		{11, "TRACK_HAND_IK_CONTROLS_1"},
		{12, "TRACK_HAND_IK_CONTROLS_2"},
		{13, "TRACK_HAND_IK_CONTROLS_3"},
		{14, "NUM_STANDARD_TRACKS"},
	};
	static const hkInternalClassEnumItem hkbGeneratorOutputTrackTypesEnumItems[] =
	{
		{0, "TRACK_TYPE_REAL"},
		{1, "TRACK_TYPE_QSTRANSFORM"},
	};
	static const hkInternalClassEnum hkbGeneratorOutputEnums[] = {
		{"StandardTracks", hkbGeneratorOutputStandardTracksEnumItems, 15, HK_NULL, 0 },
		{"TrackTypes", hkbGeneratorOutputTrackTypesEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbGeneratorOutputStandardTracksEnum = reinterpret_cast<const hkClassEnum*>(&hkbGeneratorOutputEnums[0]);
	const hkClassEnum* hkbGeneratorOutputTrackTypesEnum = reinterpret_cast<const hkClassEnum*>(&hkbGeneratorOutputEnums[1]);
	static hkInternalClassMember hkbGeneratorOutput_TrackHeaderClass_Members[] =
	{
		{ "numData", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dataOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "onFraction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "id", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "format", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, hkbGeneratorOutputTrackTypesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkbGeneratorOutputTrackHeaderClass(
		"hkbGeneratorOutputTrackHeader",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbGeneratorOutput_TrackHeaderClass_Members),
		HK_COUNT_OF(hkbGeneratorOutput_TrackHeaderClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbGeneratorOutput_TrackMasterHeaderClass_Members[] =
	{
		{ "numBytes", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numTracks", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "unused", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL }
	};
	hkClass hkbGeneratorOutputTrackMasterHeaderClass(
		"hkbGeneratorOutputTrackMasterHeader",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbGeneratorOutput_TrackMasterHeaderClass_Members),
		HK_COUNT_OF(hkbGeneratorOutput_TrackMasterHeaderClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbGeneratorOutput_TracksClass_Members[] =
	{
		{ "masterHeader", &hkbGeneratorOutputTrackMasterHeaderClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "trackHeaders", &hkbGeneratorOutputTrackHeaderClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 1, 0, 0, HK_NULL }
	};
	hkClass hkbGeneratorOutputTracksClass(
		"hkbGeneratorOutputTracks",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbGeneratorOutput_TracksClass_Members),
		HK_COUNT_OF(hkbGeneratorOutput_TracksClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbGeneratorOutput_ConstTrackClass_Members[] =
	{
		{ "header", &hkbGeneratorOutputTrackHeaderClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL }
	};
	hkClass hkbGeneratorOutputConstTrackClass(
		"hkbGeneratorOutputConstTrack",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbGeneratorOutput_ConstTrackClass_Members),
		HK_COUNT_OF(hkbGeneratorOutput_ConstTrackClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkbGeneratorOutputTrackClass(
		"hkbGeneratorOutputTrack",
		&hkbGeneratorOutputConstTrackClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbGeneratorOutputClass_Members[] =
	{
		{ "skeleton", &hkaSkeletonClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "tracks", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "deleteTracks", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbGeneratorOutputClass(
		"hkbGeneratorOutput",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbGeneratorOutputEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbGeneratorOutputClass_Members),
		HK_COUNT_OF(hkbGeneratorOutputClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkbGeneratorOutputListenerClass(
		"hkbGeneratorOutputListener",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbBehaviorReferenceGeneratorClass_Members[] =
	{
		{ "behaviorName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "behavior", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbBehaviorReferenceGeneratorClass(
		"hkbBehaviorReferenceGenerator",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbBehaviorReferenceGeneratorClass_Members),
		HK_COUNT_OF(hkbBehaviorReferenceGeneratorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbBlenderGeneratorBlenderFlagsEnumItems[] =
	{
		{1/*0x1*/, "FLAG_SYNC"},
		{4/*0x4*/, "FLAG_SMOOTH_GENERATOR_WEIGHTS"},
		{8/*0x8*/, "FLAG_DONT_DEACTIVATE_CHILDREN_WITH_ZERO_WEIGHTS"},
		{16/*0x10*/, "FLAG_PARAMETRIC_BLEND"},
		{32/*0x20*/, "FLAG_IS_PARAMETRIC_BLEND_CYCLIC"},
		{64/*0x40*/, "FLAG_FORCE_DENSE_POSE"},
	};
	static const hkInternalClassEnum hkbBlenderGeneratorEnums[] = {
		{"BlenderFlags", hkbBlenderGeneratorBlenderFlagsEnumItems, 6, HK_NULL, 0 }
	};
	const hkClassEnum* hkbBlenderGeneratorBlenderFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkbBlenderGeneratorEnums[0]);
	static hkInternalClassMember hkbBlenderGeneratorClass_Members[] =
	{
		{ "referencePoseWeightThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blendParameter", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minCyclicBlendParameter", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxCyclicBlendParameter", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "indexOfSyncMasterChild", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "flags", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "children", &hkbBlenderGeneratorChildClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "endIntervalWeight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "numActiveChildren", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "endIntervalIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "initSync", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbBlenderGenerator_DefaultStruct
		{
			int s_defaultOffsets[11];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_maxCyclicBlendParameter;
			hkInt16 m_indexOfSyncMasterChild;
		};
		const hkbBlenderGenerator_DefaultStruct hkbBlenderGenerator_Default =
		{
			{-1,-1,-1,HK_OFFSET_OF(hkbBlenderGenerator_DefaultStruct,m_maxCyclicBlendParameter),HK_OFFSET_OF(hkbBlenderGenerator_DefaultStruct,m_indexOfSyncMasterChild),-1,-1,-1,-1,-1,-1},
			1.0,-1
		};
	}
	hkClass hkbBlenderGeneratorClass(
		"hkbBlenderGenerator",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbBlenderGeneratorEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbBlenderGeneratorClass_Members),
		HK_COUNT_OF(hkbBlenderGeneratorClass_Members),
		&hkbBlenderGenerator_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbBlenderGeneratorChildOperandTypeEnumItems[] =
	{
		{0, "OPERAND_TYPE_BLEND"},
		{1, "OPERAND_TYPE_ADD"},
		{2, "OPERAND_TYPE_SUBTRACT"},
	};
	static const hkInternalClassEnum hkbBlenderGeneratorChildEnums[] = {
		{"OperandType", hkbBlenderGeneratorChildOperandTypeEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkbBlenderGeneratorChildOperandTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkbBlenderGeneratorChildEnums[0]);
	static hkInternalClassMember hkbBlenderGeneratorChildClass_Members[] =
	{
		{ "generator", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL },
		{ "weight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "worldFromModelWeight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boneWeights", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "operandType", HK_NULL, hkbBlenderGeneratorChildOperandTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "effectiveWeight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isActive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "syncNextFrame", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbBlenderGeneratorChild_DefaultStruct
		{
			int s_defaultOffsets[8];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_worldFromModelWeight;
		};
		const hkbBlenderGeneratorChild_DefaultStruct hkbBlenderGeneratorChild_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkbBlenderGeneratorChild_DefaultStruct,m_worldFromModelWeight),-1,-1,-1,-1,-1},
			1.0
		};
	}
	hkClass hkbBlenderGeneratorChildClass(
		"hkbBlenderGeneratorChild",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbBlenderGeneratorChildEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbBlenderGeneratorChildClass_Members),
		HK_COUNT_OF(hkbBlenderGeneratorChildClass_Members),
		&hkbBlenderGeneratorChild_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbClipTriggerClass_Members[] =
	{
		{ "localTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "event", &hkbEventClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "relativeToEndOfClip", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "acyclic", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isAnnotation", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbClipTriggerClass(
		"hkbClipTrigger",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbClipTriggerClass_Members),
		HK_COUNT_OF(hkbClipTriggerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbClipGeneratorPlaybackModeEnumItems[] =
	{
		{0, "MODE_SINGLE_PLAY"},
		{1, "MODE_LOOPING"},
		{2, "MODE_USER_CONTROLLED"},
		{3, "MODE_USER_CONTROLLED_LOOPING"},
		{4, "MODE_PING_PONG"},
		{5, "MODE_COUNT"},
	};
	static const hkInternalClassEnumItem hkbClipGeneratorClipFlagsEnumItems[] =
	{
		{1/*0x1*/, "FLAG_CONTINUE_MOTION_AT_END"},
		{2/*0x2*/, "FLAG_SYNC_HALF_CYCLE_IN_PING_PONG_MODE"},
		{4/*0x4*/, "FLAG_MIRROR"},
		{8/*0x8*/, "FLAG_FORCE_DENSE_POSE"},
		{16/*0x10*/, "FLAG_DONT_CONVERT_ANNOTATIONS_TO_TRIGGERS"},
	};
	static const hkInternalClassEnum hkbClipGeneratorEnums[] = {
		{"PlaybackMode", hkbClipGeneratorPlaybackModeEnumItems, 6, HK_NULL, 0 },
		{"ClipFlags", hkbClipGeneratorClipFlagsEnumItems, 5, HK_NULL, 0 }
	};
	const hkClassEnum* hkbClipGeneratorPlaybackModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbClipGeneratorEnums[0]);
	const hkClassEnum* hkbClipGeneratorClipFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkbClipGeneratorEnums[1]);
	static hkInternalClassMember hkbClipGeneratorClass_Members[] =
	{
		{ "mode", HK_NULL, hkbClipGeneratorPlaybackModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "cropStartAmountLocalTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "cropEndAmountLocalTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "startTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "playbackSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "enforcedDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "userControlledTimeFraction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "animationName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "triggers", &hkbClipTriggerClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "flags", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extractedMotion", HK_NULL, HK_NULL, hkClassMember::TYPE_QSTRANSFORM, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "echos", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "animationControl", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "previousUserControlledTimeFraction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "atEnd", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "ignoreStartTime", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "pingPongBackward", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbClipGeneratorClass(
		"hkbClipGenerator",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbClipGeneratorEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbClipGeneratorClass_Members),
		HK_COUNT_OF(hkbClipGeneratorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbManualSelectorGeneratorClass_Members[] =
	{
		{ "generators", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "selectedGeneratorIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentGeneratorIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbManualSelectorGeneratorClass(
		"hkbManualSelectorGenerator",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbManualSelectorGeneratorClass_Members),
		HK_COUNT_OF(hkbManualSelectorGeneratorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbModifierGeneratorClass_Members[] =
	{
		{ "modifier", &hkbModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "generator", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbModifierGeneratorClass(
		"hkbModifierGenerator",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbModifierGeneratorClass_Members),
		HK_COUNT_OF(hkbModifierGeneratorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbPoseMatchingGeneratorModeEnumItems[] =
	{
		{0, "MODE_MATCH"},
		{1, "MODE_PLAY"},
	};
	static const hkInternalClassEnum hkbPoseMatchingGeneratorEnums[] = {
		{"Mode", hkbPoseMatchingGeneratorModeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbPoseMatchingGeneratorModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbPoseMatchingGeneratorEnums[0]);
	static hkInternalClassMember hkbPoseMatchingGeneratorClass_Members[] =
	{
		{ "blendSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minSpeedToSwitch", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minSwitchTimeNoError", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minSwitchTimeFullError", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "startPlayingEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rootBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "otherBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "anotherBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pelvisIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "mode", HK_NULL, hkbPoseMatchingGeneratorModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "currentMatch", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "bestMatch", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeSinceBetterMatch", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "error", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "poseMatchingUtility", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbPoseMatchingGenerator_DefaultStruct
		{
			int s_defaultOffsets[15];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_blendSpeed;
			hkReal m_minSpeedToSwitch;
			hkReal m_minSwitchTimeNoError;
			hkInt32 m_startPlayingEventId;
			hkInt16 m_rootBoneIndex;
			hkInt16 m_otherBoneIndex;
			hkInt16 m_anotherBoneIndex;
			hkInt16 m_pelvisIndex;
		};
		const hkbPoseMatchingGenerator_DefaultStruct hkbPoseMatchingGenerator_Default =
		{
			{HK_OFFSET_OF(hkbPoseMatchingGenerator_DefaultStruct,m_blendSpeed),HK_OFFSET_OF(hkbPoseMatchingGenerator_DefaultStruct,m_minSpeedToSwitch),HK_OFFSET_OF(hkbPoseMatchingGenerator_DefaultStruct,m_minSwitchTimeNoError),-1,HK_OFFSET_OF(hkbPoseMatchingGenerator_DefaultStruct,m_startPlayingEventId),HK_OFFSET_OF(hkbPoseMatchingGenerator_DefaultStruct,m_rootBoneIndex),HK_OFFSET_OF(hkbPoseMatchingGenerator_DefaultStruct,m_otherBoneIndex),HK_OFFSET_OF(hkbPoseMatchingGenerator_DefaultStruct,m_anotherBoneIndex),HK_OFFSET_OF(hkbPoseMatchingGenerator_DefaultStruct,m_pelvisIndex),-1,-1,-1,-1,-1,-1},
			1.0f,0.2f,0.2f,-1,-1,-1,-1,-1
		};
	}
	hkClass hkbPoseMatchingGeneratorClass(
		"hkbPoseMatchingGenerator",
		&hkbBlenderGeneratorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbPoseMatchingGeneratorEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbPoseMatchingGeneratorClass_Members),
		HK_COUNT_OF(hkbPoseMatchingGeneratorClass_Members),
		&hkbPoseMatchingGenerator_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbPositionRelativeSelectorGeneratorClass_Members[] =
	{
		{ "registeredGenerators", &hkbRegisteredGeneratorClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "blendToFixPositionGenerator", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "autoComputeEntryPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transitionTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useCharacterForward", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "characterForward", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "positionTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fixPositionTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fixPositionEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "endFixPositionEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useManualSelection", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "selectedGeneratorIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "entryPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "entryForward", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentGeneratorIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "doLeadInFixup", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "lastTargetPos", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "lastTargetRot", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "targetLinearDisp", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "targetAngularDisp", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "clipTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "usingBlendToFixPositionGenerator", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbPositionRelativeSelectorGenerator_DefaultStruct
		{
			int s_defaultOffsets[24];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_autoComputeEntryPoints;
			hkReal m_transitionTime;
			_hkBool m_useCharacterForward;
			_hkVector4 m_characterForward;
			_hkVector4 m_targetPosition;
			_hkQuaternion m_targetRotation;
			hkReal m_positionTolerance;
			hkReal m_fixPositionTolerance;
			hkInt32 m_fixPositionEventId;
			hkInt32 m_endFixPositionEventId;
			_hkVector4 m_entryPosition;
			_hkVector4 m_entryForward;
		};
		const hkbPositionRelativeSelectorGenerator_DefaultStruct hkbPositionRelativeSelectorGenerator_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_autoComputeEntryPoints),HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_transitionTime),HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_useCharacterForward),HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_characterForward),HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_targetPosition),HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_targetRotation),HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_positionTolerance),HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_fixPositionTolerance),HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_fixPositionEventId),HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_endFixPositionEventId),-1,-1,HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_entryPosition),HK_OFFSET_OF(hkbPositionRelativeSelectorGenerator_DefaultStruct,m_entryForward),-1,-1,-1,-1,-1,-1,-1,-1},
	true,0.5,1,	{1,0,0},	{0,0,0},	{0,0,0,1},0.0001f,1.0f,-1,-1,	{0,0,0},	{1,0,0}
		};
	}
	hkClass hkbPositionRelativeSelectorGeneratorClass(
		"hkbPositionRelativeSelectorGenerator",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbPositionRelativeSelectorGeneratorClass_Members),
		HK_COUNT_OF(hkbPositionRelativeSelectorGeneratorClass_Members),
		&hkbPositionRelativeSelectorGenerator_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRegisteredGeneratorClass_Members[] =
	{
		{ "generator", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "relativePosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "relativeDirection", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbRegisteredGenerator_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_relativePosition;
			_hkVector4 m_relativeDirection;
		};
		const hkbRegisteredGenerator_DefaultStruct hkbRegisteredGenerator_Default =
		{
			{-1,HK_OFFSET_OF(hkbRegisteredGenerator_DefaultStruct,m_relativePosition),HK_OFFSET_OF(hkbRegisteredGenerator_DefaultStruct,m_relativeDirection)},
		{0,0,0},	{1,0,0}
		};
	}
	hkClass hkbRegisteredGeneratorClass(
		"hkbRegisteredGenerator",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRegisteredGeneratorClass_Members),
		HK_COUNT_OF(hkbRegisteredGeneratorClass_Members),
		&hkbRegisteredGenerator_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbReferencePoseGeneratorClass_Members[] =
	{
		{ "skeleton", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbReferencePoseGeneratorClass(
		"hkbReferencePoseGenerator",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbReferencePoseGeneratorClass_Members),
		HK_COUNT_OF(hkbReferencePoseGeneratorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbSplinePathGeneratorClass_Members[] =
	{
		{ "registeredGenerators", &hkbRegisteredGeneratorClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "characterForward", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetDirection", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "leadInGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "leadOutGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useProximityTrigger", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "endEventProximity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "endEventTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pathEndEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "selectedGeneratorIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useManualSelection", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "trackPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "usePathEstimation", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useCharacterForward", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "entryPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "entryForward", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "exitPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "exitForward", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "averageSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "pathTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "curTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "pathParam", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "currentGeneratorIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "doLeadInFixup", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "eventTriggered", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbSplinePathGenerator_DefaultStruct
		{
			int s_defaultOffsets[26];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_characterForward;
			_hkVector4 m_targetPosition;
			_hkVector4 m_targetDirection;
			hkReal m_leadInGain;
			hkReal m_leadOutGain;
			hkReal m_endEventTime;
			hkInt32 m_pathEndEventId;
			_hkBool m_trackPosition;
			_hkBool m_useCharacterForward;
		};
		const hkbSplinePathGenerator_DefaultStruct hkbSplinePathGenerator_Default =
		{
			{-1,HK_OFFSET_OF(hkbSplinePathGenerator_DefaultStruct,m_characterForward),HK_OFFSET_OF(hkbSplinePathGenerator_DefaultStruct,m_targetPosition),HK_OFFSET_OF(hkbSplinePathGenerator_DefaultStruct,m_targetDirection),HK_OFFSET_OF(hkbSplinePathGenerator_DefaultStruct,m_leadInGain),HK_OFFSET_OF(hkbSplinePathGenerator_DefaultStruct,m_leadOutGain),-1,-1,HK_OFFSET_OF(hkbSplinePathGenerator_DefaultStruct,m_endEventTime),HK_OFFSET_OF(hkbSplinePathGenerator_DefaultStruct,m_pathEndEventId),-1,-1,HK_OFFSET_OF(hkbSplinePathGenerator_DefaultStruct,m_trackPosition),-1,HK_OFFSET_OF(hkbSplinePathGenerator_DefaultStruct,m_useCharacterForward),-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
		{1,0,0},	{0,0,0},	{1,0,0},5.0f,5.0f,0.1f,-1,1,1
		};
	}
	hkClass hkbSplinePathGeneratorClass(
		"hkbSplinePathGenerator",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbSplinePathGeneratorClass_Members),
		HK_COUNT_OF(hkbSplinePathGeneratorClass_Members),
		&hkbSplinePathGenerator_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbModifierClass_Members[] =
	{
		{ "enable", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "pad", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 2, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbModifier_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_enable;
		};
		const hkbModifier_DefaultStruct hkbModifier_Default =
		{
			{HK_OFFSET_OF(hkbModifier_DefaultStruct,m_enable),-1,-1},
			true
		};
	}
	hkClass hkbModifierClass(
		"hkbModifier",
		&hkbNodeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbModifierClass_Members),
		HK_COUNT_OF(hkbModifierClass_Members),
		&hkbModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbAttachmentModifierClass_Members[] =
	{
		{ "attachmentSetup", &hkbAttachmentSetupClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "attacherHandle", &hkbHandleClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "attacheeHandle", &hkbHandleClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "sendToAttacherOnAttach", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sendToAttacheeOnAttach", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sendToAttacherOnDetach", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sendToAttacheeOnDetach", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "attacheeLayer", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "attacheeRB", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "oldFilterInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "attachment", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbAttachmentModifier_DefaultStruct
		{
			int s_defaultOffsets[11];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_sendToAttacherOnAttach;
			hkInt32 m_sendToAttacheeOnAttach;
			hkInt32 m_sendToAttacherOnDetach;
			hkInt32 m_sendToAttacheeOnDetach;
			hkInt32 m_attacheeLayer;
		};
		const hkbAttachmentModifier_DefaultStruct hkbAttachmentModifier_Default =
		{
			{-1,-1,-1,HK_OFFSET_OF(hkbAttachmentModifier_DefaultStruct,m_sendToAttacherOnAttach),HK_OFFSET_OF(hkbAttachmentModifier_DefaultStruct,m_sendToAttacheeOnAttach),HK_OFFSET_OF(hkbAttachmentModifier_DefaultStruct,m_sendToAttacherOnDetach),HK_OFFSET_OF(hkbAttachmentModifier_DefaultStruct,m_sendToAttacheeOnDetach),HK_OFFSET_OF(hkbAttachmentModifier_DefaultStruct,m_attacheeLayer),-1,-1,-1},
			-1,-1,-1,-1,-1
		};
	}
	hkClass hkbAttachmentModifierClass(
		"hkbAttachmentModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbAttachmentModifierClass_Members),
		HK_COUNT_OF(hkbAttachmentModifierClass_Members),
		&hkbAttachmentModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbAttributeModifier_AssignmentClass_Members[] =
	{
		{ "attributeIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "attributeValue", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbAttributeModifierAssignment_DefaultStruct
		{
			int s_defaultOffsets[2];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_attributeIndex;
		};
		const hkbAttributeModifierAssignment_DefaultStruct hkbAttributeModifierAssignment_Default =
		{
			{HK_OFFSET_OF(hkbAttributeModifierAssignment_DefaultStruct,m_attributeIndex),-1},
			-1
		};
	}
	hkClass hkbAttributeModifierAssignmentClass(
		"hkbAttributeModifierAssignment",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbAttributeModifier_AssignmentClass_Members),
		HK_COUNT_OF(hkbAttributeModifier_AssignmentClass_Members),
		&hkbAttributeModifierAssignment_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbAttributeModifierClass_Members[] =
	{
		{ "assignments", &hkbAttributeModifierAssignmentClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbAttributeModifierClass(
		"hkbAttributeModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbAttributeModifierClass_Members),
		HK_COUNT_OF(hkbAttributeModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbComputeDirectionModifierClass_Members[] =
	{
		{ "pointIn", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pointOut", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "groundAngleOut", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "upAngleOut", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "verticalOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "reverseGroundAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "reverseUpAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "projectPoint", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normalizePoint", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "computeOnlyOnce", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "computedOutput", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbComputeDirectionModifierClass(
		"hkbComputeDirectionModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbComputeDirectionModifierClass_Members),
		HK_COUNT_OF(hkbComputeDirectionModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbDampingModifierClass_Members[] =
	{
		{ "kP", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "kI", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "kD", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "errorSum", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "previousError", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rawValue", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dampedValue", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbDampingModifier_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_kP;
			hkReal m_kI;
			hkReal m_kD;
		};
		const hkbDampingModifier_DefaultStruct hkbDampingModifier_Default =
		{
			{HK_OFFSET_OF(hkbDampingModifier_DefaultStruct,m_kP),HK_OFFSET_OF(hkbDampingModifier_DefaultStruct,m_kI),HK_OFFSET_OF(hkbDampingModifier_DefaultStruct,m_kD),-1,-1,-1,-1},
			0.20f,0.015f,-0.10f
		};
	}
	hkClass hkbDampingModifierClass(
		"hkbDampingModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbDampingModifierClass_Members),
		HK_COUNT_OF(hkbDampingModifierClass_Members),
		&hkbDampingModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbDelayedModifierClass_Members[] =
	{
		{ "delaySeconds", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "durationSeconds", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "secondsElapsed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isActive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbDelayedModifierClass(
		"hkbDelayedModifier",
		&hkbModifierWrapperClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbDelayedModifierClass_Members),
		HK_COUNT_OF(hkbDelayedModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbEvaluateHandleModifierClass_Members[] =
	{
		{ "handle", &hkbHandleClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "handlePositionOut", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "handleRotationOut", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isValidOut", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbEvaluateHandleModifier_DefaultStruct
		{
			int s_defaultOffsets[4];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_handlePositionOut;
			_hkQuaternion m_handleRotationOut;
		};
		const hkbEvaluateHandleModifier_DefaultStruct hkbEvaluateHandleModifier_Default =
		{
			{-1,HK_OFFSET_OF(hkbEvaluateHandleModifier_DefaultStruct,m_handlePositionOut),HK_OFFSET_OF(hkbEvaluateHandleModifier_DefaultStruct,m_handleRotationOut),-1},
		{0,0,0},	{0,0,0,1}
		};
	}
	hkClass hkbEvaluateHandleModifierClass(
		"hkbEvaluateHandleModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbEvaluateHandleModifierClass_Members),
		HK_COUNT_OF(hkbEvaluateHandleModifierClass_Members),
		&hkbEvaluateHandleModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbEventDrivenModifierClass_Members[] =
	{
		{ "activateEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deactivateEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "activeByDefault", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isActive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbEventDrivenModifier_DefaultStruct
		{
			int s_defaultOffsets[4];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_activateEventId;
			hkInt32 m_deactivateEventId;
		};
		const hkbEventDrivenModifier_DefaultStruct hkbEventDrivenModifier_Default =
		{
			{HK_OFFSET_OF(hkbEventDrivenModifier_DefaultStruct,m_activateEventId),HK_OFFSET_OF(hkbEventDrivenModifier_DefaultStruct,m_deactivateEventId),-1,-1},
			-1,-1
		};
	}
	hkClass hkbEventDrivenModifierClass(
		"hkbEventDrivenModifier",
		&hkbModifierWrapperClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbEventDrivenModifierClass_Members),
		HK_COUNT_OF(hkbEventDrivenModifierClass_Members),
		&hkbEventDrivenModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbFootIkControlDataClass_Members[] =
	{
		{ "gains", &hkbFootIkGainsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkbFootIkControlDataClass(
		"hkbFootIkControlData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbFootIkControlDataClass_Members),
		HK_COUNT_OF(hkbFootIkControlDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbFootIkControlsModifierClass_Members[] =
	{
		{ "controlData", &hkbFootIkControlDataClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbFootIkControlsModifierClass(
		"hkbFootIkControlsModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbFootIkControlsModifierClass_Members),
		HK_COUNT_OF(hkbFootIkControlsModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbFootIkGainsClass_Members[] =
	{
		{ "onOffGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "groundAscendingGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "groundDescendingGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "footPlantedGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "footRaisedGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "footUnlockGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "worldFromModelFeedbackGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "errorUpDownBias", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "alignWorldFromModelGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hipOrientationGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbFootIkGains_DefaultStruct
		{
			int s_defaultOffsets[10];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_onOffGain;
			hkReal m_groundAscendingGain;
			hkReal m_groundDescendingGain;
			hkReal m_footPlantedGain;
			hkReal m_footRaisedGain;
			hkReal m_footUnlockGain;
			hkReal m_errorUpDownBias;
		};
		const hkbFootIkGains_DefaultStruct hkbFootIkGains_Default =
		{
			{HK_OFFSET_OF(hkbFootIkGains_DefaultStruct,m_onOffGain),HK_OFFSET_OF(hkbFootIkGains_DefaultStruct,m_groundAscendingGain),HK_OFFSET_OF(hkbFootIkGains_DefaultStruct,m_groundDescendingGain),HK_OFFSET_OF(hkbFootIkGains_DefaultStruct,m_footPlantedGain),HK_OFFSET_OF(hkbFootIkGains_DefaultStruct,m_footRaisedGain),HK_OFFSET_OF(hkbFootIkGains_DefaultStruct,m_footUnlockGain),-1,HK_OFFSET_OF(hkbFootIkGains_DefaultStruct,m_errorUpDownBias),-1,-1},
			0.2f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f
		};
	}
	hkClass hkbFootIkGainsClass(
		"hkbFootIkGains",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbFootIkGainsClass_Members),
		HK_COUNT_OF(hkbFootIkGainsClass_Members),
		&hkbFootIkGains_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbFootIkModifier_LegClass_Members[] =
	{
		{ "originalAnkleTransformMS", HK_NULL, HK_NULL, hkClassMember::TYPE_QSTRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "kneeAxisLS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "footEndLS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "footPlantedAnkleHeightMS", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "footRaisedAnkleHeightMS", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxAnkleHeightMS", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minAnkleHeightMS", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxKneeAngleDegrees", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minKneeAngleDegrees", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ungroundedEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "legIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hipIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "kneeIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ankleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isOriginalAnkleTransformMSSet", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbFootIkModifierLeg_DefaultStruct
		{
			int s_defaultOffsets[15];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkQsTransform m_originalAnkleTransformMS;
			_hkVector4 m_kneeAxisLS;
			_hkVector4 m_footEndLS;
			hkReal m_footRaisedAnkleHeightMS;
			hkReal m_maxAnkleHeightMS;
			hkReal m_minAnkleHeightMS;
			hkReal m_maxKneeAngleDegrees;
			hkInt32 m_ungroundedEventId;
			hkInt16 m_legIndex;
			hkInt16 m_hipIndex;
			hkInt16 m_kneeIndex;
			hkInt16 m_ankleIndex;
		};
		const hkbFootIkModifierLeg_DefaultStruct hkbFootIkModifierLeg_Default =
		{
			{HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_originalAnkleTransformMS),HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_kneeAxisLS),HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_footEndLS),-1,HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_footRaisedAnkleHeightMS),HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_maxAnkleHeightMS),HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_minAnkleHeightMS),HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_maxKneeAngleDegrees),-1,HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_ungroundedEventId),HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_legIndex),HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_hipIndex),HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_kneeIndex),HK_OFFSET_OF(hkbFootIkModifierLeg_DefaultStruct,m_ankleIndex),-1},
		{0,0,0,0,0,0,0,1,1,1,1,1},	{1,0,0},	{0,0,0},0.5f,0.7f,-0.1f,180.0f,-1,-1,-1,-1,-1
		};
	}
	hkClass hkbFootIkModifierLegClass(
		"hkbFootIkModifierLeg",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbFootIkModifier_LegClass_Members),
		HK_COUNT_OF(hkbFootIkModifier_LegClass_Members),
		&hkbFootIkModifierLeg_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbFootIkModifier_InternalLegDataClass_Members[] =
	{
		{ "groundPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "footIkSolver", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "verticalError", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hitSomething", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isPlanted", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbFootIkModifierInternalLegDataClass(
		"hkbFootIkModifierInternalLegData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbFootIkModifier_InternalLegDataClass_Members),
		HK_COUNT_OF(hkbFootIkModifier_InternalLegDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbFootIkModifierClass_Members[] =
	{
		{ "gains", &hkbFootIkGainsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "legs", &hkbFootIkModifierLegClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "raycastDistanceUp", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "raycastDistanceDown", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "originalGroundHeightMS", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "errorOut", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionFilterInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useTrackData", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lockFeetWhenPlanted", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useCharacterUpVector", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "raycastInterface", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "internalLegData", &hkbFootIkModifierInternalLegDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "prevIsFootIkEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isSetUp", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isGroundPositionValid", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbFootIkModifier_DefaultStruct
		{
			int s_defaultOffsets[15];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_raycastDistanceUp;
			hkReal m_raycastDistanceDown;
			hkUint32 m_collisionFilterInfo;
		};
		const hkbFootIkModifier_DefaultStruct hkbFootIkModifier_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkbFootIkModifier_DefaultStruct,m_raycastDistanceUp),HK_OFFSET_OF(hkbFootIkModifier_DefaultStruct,m_raycastDistanceDown),-1,-1,HK_OFFSET_OF(hkbFootIkModifier_DefaultStruct,m_collisionFilterInfo),-1,-1,-1,-1,-1,-1,-1,-1},
			0.5f,0.8f,3
		};
	}
	hkClass hkbFootIkModifierClass(
		"hkbFootIkModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbFootIkModifierClass_Members),
		HK_COUNT_OF(hkbFootIkModifierClass_Members),
		&hkbFootIkModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbGetHandleOnBoneModifierClass_Members[] =
	{
		{ "handle", &hkbHandleClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "handleOut", &hkbHandleClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "localFrameName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "animationBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbGetHandleOnBoneModifier_DefaultStruct
		{
			int s_defaultOffsets[5];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt16 m_ragdollBoneIndex;
			hkInt16 m_animationBoneIndex;
		};
		const hkbGetHandleOnBoneModifier_DefaultStruct hkbGetHandleOnBoneModifier_Default =
		{
			{-1,-1,-1,HK_OFFSET_OF(hkbGetHandleOnBoneModifier_DefaultStruct,m_ragdollBoneIndex),HK_OFFSET_OF(hkbGetHandleOnBoneModifier_DefaultStruct,m_animationBoneIndex)},
			-1,-1
		};
	}
	hkClass hkbGetHandleOnBoneModifierClass(
		"hkbGetHandleOnBoneModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbGetHandleOnBoneModifierClass_Members),
		HK_COUNT_OF(hkbGetHandleOnBoneModifierClass_Members),
		&hkbGetHandleOnBoneModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbGetUpModifierClass_Members[] =
	{
		{ "duration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rootBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "otherBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "anotherBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initNextModify", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeSinceBegin", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeStep", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbGetUpModifier_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_duration;
			hkInt16 m_rootBoneIndex;
			hkInt16 m_otherBoneIndex;
			hkInt16 m_anotherBoneIndex;
		};
		const hkbGetUpModifier_DefaultStruct hkbGetUpModifier_Default =
		{
			{HK_OFFSET_OF(hkbGetUpModifier_DefaultStruct,m_duration),HK_OFFSET_OF(hkbGetUpModifier_DefaultStruct,m_rootBoneIndex),HK_OFFSET_OF(hkbGetUpModifier_DefaultStruct,m_otherBoneIndex),HK_OFFSET_OF(hkbGetUpModifier_DefaultStruct,m_anotherBoneIndex),-1,-1,-1},
			1.0,-1,-1,-1
		};
	}
	hkClass hkbGetUpModifierClass(
		"hkbGetUpModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbGetUpModifierClass_Members),
		HK_COUNT_OF(hkbGetUpModifierClass_Members),
		&hkbGetUpModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbHandIkControlDataClass_Members[] =
	{
		{ "targetPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetNormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "positionOnFraction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normalOnFraction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeInDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeOutDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbHandIkControlData_DefaultStruct
		{
			int s_defaultOffsets[6];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_targetPosition;
			_hkVector4 m_targetNormal;
		};
		const hkbHandIkControlData_DefaultStruct hkbHandIkControlData_Default =
		{
			{HK_OFFSET_OF(hkbHandIkControlData_DefaultStruct,m_targetPosition),HK_OFFSET_OF(hkbHandIkControlData_DefaultStruct,m_targetNormal),-1,-1,-1,-1},
		{0,0,0,0},	{0,0,0}
		};
	}
	hkClass hkbHandIkControlDataClass(
		"hkbHandIkControlData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbHandIkControlDataClass_Members),
		HK_COUNT_OF(hkbHandIkControlDataClass_Members),
		&hkbHandIkControlData_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbHandIkControlsModifier_HandClass_Members[] =
	{
		{ "controlData", &hkbHandIkControlDataClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "handIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "enable", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbHandIkControlsModifierHandClass(
		"hkbHandIkControlsModifierHand",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbHandIkControlsModifier_HandClass_Members),
		HK_COUNT_OF(hkbHandIkControlsModifier_HandClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbHandIkControlsModifierClass_Members[] =
	{
		{ "hands", &hkbHandIkControlsModifierHandClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbHandIkControlsModifierClass(
		"hkbHandIkControlsModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbHandIkControlsModifierClass_Members),
		HK_COUNT_OF(hkbHandIkControlsModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbHandIkModifier_HandClass_Members[] =
	{
		{ "elbowAxisLS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "backHandNormalLS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxElbowAngleDegrees", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minElbowAngleDegrees", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "handIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "shoulderIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "shoulderSiblingIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "elbowIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "elbowSiblingIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wristIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbHandIkModifierHand_DefaultStruct
		{
			int s_defaultOffsets[10];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_elbowAxisLS;
			_hkVector4 m_backHandNormalLS;
			hkReal m_maxElbowAngleDegrees;
			hkInt16 m_handIndex;
			hkInt16 m_shoulderIndex;
			hkInt16 m_shoulderSiblingIndex;
			hkInt16 m_elbowIndex;
			hkInt16 m_elbowSiblingIndex;
			hkInt16 m_wristIndex;
		};
		const hkbHandIkModifierHand_DefaultStruct hkbHandIkModifierHand_Default =
		{
			{HK_OFFSET_OF(hkbHandIkModifierHand_DefaultStruct,m_elbowAxisLS),HK_OFFSET_OF(hkbHandIkModifierHand_DefaultStruct,m_backHandNormalLS),HK_OFFSET_OF(hkbHandIkModifierHand_DefaultStruct,m_maxElbowAngleDegrees),-1,HK_OFFSET_OF(hkbHandIkModifierHand_DefaultStruct,m_handIndex),HK_OFFSET_OF(hkbHandIkModifierHand_DefaultStruct,m_shoulderIndex),HK_OFFSET_OF(hkbHandIkModifierHand_DefaultStruct,m_shoulderSiblingIndex),HK_OFFSET_OF(hkbHandIkModifierHand_DefaultStruct,m_elbowIndex),HK_OFFSET_OF(hkbHandIkModifierHand_DefaultStruct,m_elbowSiblingIndex),HK_OFFSET_OF(hkbHandIkModifierHand_DefaultStruct,m_wristIndex)},
		{0,0,0},	{0,0,0},180.0f,-1,-1,-1,-1,-1,-1
		};
	}
	hkClass hkbHandIkModifierHandClass(
		"hkbHandIkModifierHand",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbHandIkModifier_HandClass_Members),
		HK_COUNT_OF(hkbHandIkModifier_HandClass_Members),
		&hkbHandIkModifierHand_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbHandIkModifierClass_Members[] =
	{
		{ "hands", &hkbHandIkModifierHandClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "fadeInOutCurve", HK_NULL, hkbBlendCurveUtilsBlendCurveEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "internalHandData", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbHandIkModifierClass(
		"hkbHandIkModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbHandIkModifierClass_Members),
		HK_COUNT_OF(hkbHandIkModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbJigglerGroupClass_Members[] =
	{
		{ "boneIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "mass", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stiffness", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxElongation", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxCompression", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "propagateToChildren", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "affectSiblings", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotateBonesForSkinning", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentVelocitiesWS", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "currentPositionsWS", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbJigglerGroup_DefaultStruct
		{
			int s_defaultOffsets[11];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_mass;
			hkReal m_stiffness;
			hkReal m_damping;
			hkReal m_maxElongation;
			hkReal m_maxCompression;
			_hkBool m_propagateToChildren;
		};
		const hkbJigglerGroup_DefaultStruct hkbJigglerGroup_Default =
		{
			{-1,HK_OFFSET_OF(hkbJigglerGroup_DefaultStruct,m_mass),HK_OFFSET_OF(hkbJigglerGroup_DefaultStruct,m_stiffness),HK_OFFSET_OF(hkbJigglerGroup_DefaultStruct,m_damping),HK_OFFSET_OF(hkbJigglerGroup_DefaultStruct,m_maxElongation),HK_OFFSET_OF(hkbJigglerGroup_DefaultStruct,m_maxCompression),HK_OFFSET_OF(hkbJigglerGroup_DefaultStruct,m_propagateToChildren),-1,-1,-1,-1},
			1.0f,100.0f,1.0f,1.0f,0.5f,true
		};
	}
	hkClass hkbJigglerGroupClass(
		"hkbJigglerGroup",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbJigglerGroupClass_Members),
		HK_COUNT_OF(hkbJigglerGroupClass_Members),
		&hkbJigglerGroup_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbJigglerModifierClass_Members[] =
	{
		{ "jigglerGroups", &hkbJigglerGroupClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "timeStep", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "initNextModify", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbJigglerModifier_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_initNextModify;
		};
		const hkbJigglerModifier_DefaultStruct hkbJigglerModifier_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkbJigglerModifier_DefaultStruct,m_initNextModify)},
			true
		};
	}
	hkClass hkbJigglerModifierClass(
		"hkbJigglerModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbJigglerModifierClass_Members),
		HK_COUNT_OF(hkbJigglerModifierClass_Members),
		&hkbJigglerModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbModifierListClass_Members[] =
	{
		{ "modifiers", &hkbModifierClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkbModifierListClass(
		"hkbModifierList",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbModifierListClass_Members),
		HK_COUNT_OF(hkbModifierListClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbLookAtModifierClass_Members[] =
	{
		{ "newTargetGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "onGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "offGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "limitAngleDegrees", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetWS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lookUp", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lookUpAngleDegrees", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "headForwardHS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "headRightHS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isOn", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "headIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "neckIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lookAtLastTargetWS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "lookAtWeight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbLookAtModifier_DefaultStruct
		{
			int s_defaultOffsets[14];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_newTargetGain;
			hkReal m_onGain;
			hkReal m_offGain;
			hkReal m_limitAngleDegrees;
			_hkVector4 m_targetWS;
			_hkVector4 m_headForwardHS;
			_hkVector4 m_headRightHS;
			_hkBool m_isOn;
			hkInt16 m_headIndex;
			hkInt16 m_neckIndex;
		};
		const hkbLookAtModifier_DefaultStruct hkbLookAtModifier_Default =
		{
			{HK_OFFSET_OF(hkbLookAtModifier_DefaultStruct,m_newTargetGain),HK_OFFSET_OF(hkbLookAtModifier_DefaultStruct,m_onGain),HK_OFFSET_OF(hkbLookAtModifier_DefaultStruct,m_offGain),HK_OFFSET_OF(hkbLookAtModifier_DefaultStruct,m_limitAngleDegrees),HK_OFFSET_OF(hkbLookAtModifier_DefaultStruct,m_targetWS),-1,-1,HK_OFFSET_OF(hkbLookAtModifier_DefaultStruct,m_headForwardHS),HK_OFFSET_OF(hkbLookAtModifier_DefaultStruct,m_headRightHS),HK_OFFSET_OF(hkbLookAtModifier_DefaultStruct,m_isOn),HK_OFFSET_OF(hkbLookAtModifier_DefaultStruct,m_headIndex),HK_OFFSET_OF(hkbLookAtModifier_DefaultStruct,m_neckIndex),-1,-1},
	0.2f,0.05f,0.05f,45.0,	{0,0,0},	{0,1,0},	{1,0,0},true,-1,-1
		};
	}
	hkClass hkbLookAtModifierClass(
		"hkbLookAtModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbLookAtModifierClass_Members),
		HK_COUNT_OF(hkbLookAtModifierClass_Members),
		&hkbLookAtModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbMirrorModifierClass_Members[] =
	{
		{ "isAdditive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbMirrorModifierClass(
		"hkbMirrorModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbMirrorModifierClass_Members),
		HK_COUNT_OF(hkbMirrorModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbMoveCharacterModifierClass_Members[] =
	{
		{ "offsetPerSecondMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "timeSinceLastModify", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbMoveCharacterModifier_DefaultStruct
		{
			int s_defaultOffsets[2];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_offsetPerSecondMS;
		};
		const hkbMoveCharacterModifier_DefaultStruct hkbMoveCharacterModifier_Default =
		{
			{HK_OFFSET_OF(hkbMoveCharacterModifier_DefaultStruct,m_offsetPerSecondMS),-1},
			{0,0,0}
		};
	}
	hkClass hkbMoveCharacterModifierClass(
		"hkbMoveCharacterModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbMoveCharacterModifierClass_Members),
		HK_COUNT_OF(hkbMoveCharacterModifierClass_Members),
		&hkbMoveCharacterModifier_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbProxyModifierPhantomTypeEnumItems[] =
	{
		{0, "PHANTOM_TYPE_SIMPLE"},
		{1, "PHANTOM_TYPE_CACHING"},
	};
	static const hkInternalClassEnumItem hkbProxyModifierLinearVelocityModeEnumItems[] =
	{
		{0, "LINEAR_VELOCITY_MODE_WORLD"},
		{1, "LINEAR_VELOCITY_MODE_MODEL"},
	};
	static const hkInternalClassEnum hkbProxyModifierEnums[] = {
		{"PhantomType", hkbProxyModifierPhantomTypeEnumItems, 2, HK_NULL, 0 },
		{"LinearVelocityMode", hkbProxyModifierLinearVelocityModeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbProxyModifierPhantomTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkbProxyModifierEnums[0]);
	const hkClassEnum* hkbProxyModifierLinearVelocityModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbProxyModifierEnums[1]);
	static hkInternalClassMember hkbProxyModifier_ProxyInfoClass_Members[] =
	{
		{ "dynamicFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "staticFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "keepContactTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "up", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "keepDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contactAngleSensitivity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "userPlanes", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxCharacterSpeedForSolver", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "characterStrength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "characterMass", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxSlope", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "penetrationRecoverySpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxCastIterations", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "refreshManifoldInCheckSupport", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbProxyModifierProxyInfo_DefaultStruct
		{
			int s_defaultOffsets[14];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_dynamicFriction;
			hkReal m_keepContactTolerance;
			_hkVector4 m_up;
			hkReal m_keepDistance;
			hkReal m_contactAngleSensitivity;
			hkUint32 m_userPlanes;
			hkReal m_maxCharacterSpeedForSolver;
			hkReal m_characterStrength;
			hkReal m_maxSlope;
			hkReal m_penetrationRecoverySpeed;
			int m_maxCastIterations;
		};
		const hkbProxyModifierProxyInfo_DefaultStruct hkbProxyModifierProxyInfo_Default =
		{
			{HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_dynamicFriction),-1,HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_keepContactTolerance),HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_up),HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_keepDistance),HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_contactAngleSensitivity),HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_userPlanes),HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_maxCharacterSpeedForSolver),HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_characterStrength),-1,HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_maxSlope),HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_penetrationRecoverySpeed),HK_OFFSET_OF(hkbProxyModifierProxyInfo_DefaultStruct,m_maxCastIterations),-1},
		1.0f,0.1f,	{0,1,0},0.05f,10,4,10,7.9E+28f,90.0,1.0f,10
		};
	}
	hkClass hkbProxyModifierProxyInfoClass(
		"hkbProxyModifierProxyInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbProxyModifier_ProxyInfoClass_Members),
		HK_COUNT_OF(hkbProxyModifier_ProxyInfoClass_Members),
		&hkbProxyModifierProxyInfo_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbProxyModifierClass_Members[] =
	{
		{ "proxyInfo", &hkbProxyModifierProxyInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "linearVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "horizontalGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "verticalGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxHorizontalSeparation", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxVerticalSeparation", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "verticalDisplacementError", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "verticalDisplacementErrorGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxVerticalDisplacement", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minVerticalDisplacement", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "capsuleHeight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "capsuleRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxSlopeForRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionFilterInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "phantomType", HK_NULL, hkbProxyModifierPhantomTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "linearVelocityMode", HK_NULL, hkbProxyModifierLinearVelocityModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "ignoreIncomingRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ignoreCollisionDuringRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ignoreIncomingTranslation", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "characterProxy", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "phantom", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "phantomShape", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "horizontalDisplacement", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "verticalDisplacement", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timestep", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbProxyModifier_DefaultStruct
		{
			int s_defaultOffsets[25];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_linearVelocity;
			hkReal m_horizontalGain;
			hkReal m_verticalGain;
			hkReal m_maxHorizontalSeparation;
			hkReal m_maxVerticalSeparation;
			hkReal m_maxVerticalDisplacement;
			hkReal m_minVerticalDisplacement;
			hkReal m_capsuleHeight;
			hkReal m_capsuleRadius;
			hkReal m_maxSlopeForRotation;
			_hkBool m_ignoreIncomingRotation;
			_hkBool m_ignoreCollisionDuringRotation;
		};
		const hkbProxyModifier_DefaultStruct hkbProxyModifier_Default =
		{
			{-1,HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_linearVelocity),HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_horizontalGain),HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_verticalGain),HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_maxHorizontalSeparation),HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_maxVerticalSeparation),-1,-1,HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_maxVerticalDisplacement),HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_minVerticalDisplacement),HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_capsuleHeight),HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_capsuleRadius),HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_maxSlopeForRotation),-1,-1,-1,HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_ignoreIncomingRotation),HK_OFFSET_OF(hkbProxyModifier_DefaultStruct,m_ignoreCollisionDuringRotation),-1,-1,-1,-1,-1,-1,-1},
			{0,0,0},0.8f,0.2f,0.15f,5.0f,0.5f,-0.5f,1.7f,0.4f,90.0,true,true
		};
	}
	hkClass hkbProxyModifierClass(
		"hkbProxyModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbProxyModifierEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbProxyModifierClass_Members),
		HK_COUNT_OF(hkbProxyModifierClass_Members),
		&hkbProxyModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbKeyframeBonesModifierClass_Members[] =
	{
		{ "keyframedBonesList", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "keyframedBones", &hkBitFieldClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbKeyframeBonesModifierClass(
		"hkbKeyframeBonesModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbKeyframeBonesModifierClass_Members),
		HK_COUNT_OF(hkbKeyframeBonesModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbPoweredRagdollControlDataClass_Members[] =
	{
		{ "maxForce", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL },
		{ "tau", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "proportionalRecoveryVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "constantRecoveryVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbPoweredRagdollControlData_DefaultStruct
		{
			int s_defaultOffsets[5];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_maxForce;
			hkReal m_tau;
			hkReal m_damping;
			hkReal m_proportionalRecoveryVelocity;
			hkReal m_constantRecoveryVelocity;
		};
		const hkbPoweredRagdollControlData_DefaultStruct hkbPoweredRagdollControlData_Default =
		{
			{HK_OFFSET_OF(hkbPoweredRagdollControlData_DefaultStruct,m_maxForce),HK_OFFSET_OF(hkbPoweredRagdollControlData_DefaultStruct,m_tau),HK_OFFSET_OF(hkbPoweredRagdollControlData_DefaultStruct,m_damping),HK_OFFSET_OF(hkbPoweredRagdollControlData_DefaultStruct,m_proportionalRecoveryVelocity),HK_OFFSET_OF(hkbPoweredRagdollControlData_DefaultStruct,m_constantRecoveryVelocity)},
			200.0f,0.8f,1.0f,2.0f,1.0f
		};
	}
	hkClass hkbPoweredRagdollControlDataClass(
		"hkbPoweredRagdollControlData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbPoweredRagdollControlDataClass_Members),
		HK_COUNT_OF(hkbPoweredRagdollControlDataClass_Members),
		&hkbPoweredRagdollControlData_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbPoweredRagdollControlsModifierClass_Members[] =
	{
		{ "controlData", &hkbPoweredRagdollControlDataClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boneWeights", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL }
	};
	hkClass hkbPoweredRagdollControlsModifierClass(
		"hkbPoweredRagdollControlsModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbPoweredRagdollControlsModifierClass_Members),
		HK_COUNT_OF(hkbPoweredRagdollControlsModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbPoweredRagdollModifierComputeWorldFromModelModeEnumItems[] =
	{
		{0, "WORLD_FROM_MODEL_MODE_COMPUTE"},
		{1, "WORLD_FROM_MODEL_MODE_USE_INOUT"},
		{2, "WORLD_FROM_MODEL_MODE_USE_INPUT"},
	};
	static const hkInternalClassEnum hkbPoweredRagdollModifierEnums[] = {
		{"ComputeWorldFromModelMode", hkbPoweredRagdollModifierComputeWorldFromModelModeEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkbPoweredRagdollModifierComputeWorldFromModelModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbPoweredRagdollModifierEnums[0]);
	static hkInternalClassMember hkbPoweredRagdollModifier_KeyframeInfoClass_Members[] =
	{
		{ "keyframedPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "keyframedRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isValid", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isValidOut", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbPoweredRagdollModifierKeyframeInfo_DefaultStruct
		{
			int s_defaultOffsets[5];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_keyframedPosition;
			_hkQuaternion m_keyframedRotation;
			hkInt16 m_boneIndex;
		};
		const hkbPoweredRagdollModifierKeyframeInfo_DefaultStruct hkbPoweredRagdollModifierKeyframeInfo_Default =
		{
			{HK_OFFSET_OF(hkbPoweredRagdollModifierKeyframeInfo_DefaultStruct,m_keyframedPosition),HK_OFFSET_OF(hkbPoweredRagdollModifierKeyframeInfo_DefaultStruct,m_keyframedRotation),HK_OFFSET_OF(hkbPoweredRagdollModifierKeyframeInfo_DefaultStruct,m_boneIndex),-1,-1},
		{0,0,0},	{0,0,0,1},-1
		};
	}
	hkClass hkbPoweredRagdollModifierKeyframeInfoClass(
		"hkbPoweredRagdollModifierKeyframeInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbPoweredRagdollModifier_KeyframeInfoClass_Members),
		HK_COUNT_OF(hkbPoweredRagdollModifier_KeyframeInfoClass_Members),
		&hkbPoweredRagdollModifierKeyframeInfo_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbPoweredRagdollModifierClass_Members[] =
	{
		{ "floorRaycastLayer", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "controls", &hkbPoweredRagdollControlDataClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blendInTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "computeWorldFromModelMode", HK_NULL, hkbPoweredRagdollModifierComputeWorldFromModelModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "fixConstraintsTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useLocking", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "keyframeInfo", &hkbPoweredRagdollModifierKeyframeInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "rootBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "otherBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "anotherBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "timeActive", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeSinceLastModify", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "keyframedBones", &hkBitFieldClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boneWeights", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "boneIndexToKeyframeInfoIndexMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbPoweredRagdollModifier_DefaultStruct
		{
			int s_defaultOffsets[15];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_floorRaycastLayer;
			hkReal m_blendInTime;
			hkReal m_fixConstraintsTime;
			hkInt16 m_rootBoneIndex;
			hkInt16 m_otherBoneIndex;
			hkInt16 m_anotherBoneIndex;
		};
		const hkbPoweredRagdollModifier_DefaultStruct hkbPoweredRagdollModifier_Default =
		{
			{HK_OFFSET_OF(hkbPoweredRagdollModifier_DefaultStruct,m_floorRaycastLayer),-1,HK_OFFSET_OF(hkbPoweredRagdollModifier_DefaultStruct,m_blendInTime),-1,HK_OFFSET_OF(hkbPoweredRagdollModifier_DefaultStruct,m_fixConstraintsTime),-1,-1,HK_OFFSET_OF(hkbPoweredRagdollModifier_DefaultStruct,m_rootBoneIndex),HK_OFFSET_OF(hkbPoweredRagdollModifier_DefaultStruct,m_otherBoneIndex),HK_OFFSET_OF(hkbPoweredRagdollModifier_DefaultStruct,m_anotherBoneIndex),-1,-1,-1,-1,-1},
			-1,0.2f,0.5f,-1,-1,-1
		};
	}
	hkClass hkbPoweredRagdollModifierClass(
		"hkbPoweredRagdollModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbPoweredRagdollModifierEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbPoweredRagdollModifierClass_Members),
		HK_COUNT_OF(hkbPoweredRagdollModifierClass_Members),
		&hkbPoweredRagdollModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRagdollDriverModifierClass_Members[] =
	{
		{ "addRagdollToWorld", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "removeRagdollFromWorld", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "poweredRagdollModifier", &hkbPoweredRagdollModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "rigidBodyRagdollModifier", &hkbRigidBodyRagdollModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "ragdollForceModifier", &hkbRagdollForceModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "activeModifier", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isRagdollForceModifierActive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "doSetup", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbRagdollDriverModifier_DefaultStruct
		{
			int s_defaultOffsets[8];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_doSetup;
		};
		const hkbRagdollDriverModifier_DefaultStruct hkbRagdollDriverModifier_Default =
		{
			{-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkbRagdollDriverModifier_DefaultStruct,m_doSetup)},
			true
		};
	}
	hkClass hkbRagdollDriverModifierClass(
		"hkbRagdollDriverModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRagdollDriverModifierClass_Members),
		HK_COUNT_OF(hkbRagdollDriverModifierClass_Members),
		&hkbRagdollDriverModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRagdollForceModifierClass_Members[] =
	{
		{ "boneForces", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL }
	};
	hkClass hkbRagdollForceModifierClass(
		"hkbRagdollForceModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRagdollForceModifierClass_Members),
		HK_COUNT_OF(hkbRagdollForceModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRigidBodyRagdollControlDataClass_Members[] =
	{
		{ "keyFrameHierarchyControlData", &hkaKeyFrameHierarchyUtilityControlDataClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL },
		{ "durationToBlend", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbRigidBodyRagdollControlData_DefaultStruct
		{
			int s_defaultOffsets[2];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_durationToBlend;
		};
		const hkbRigidBodyRagdollControlData_DefaultStruct hkbRigidBodyRagdollControlData_Default =
		{
			{-1,HK_OFFSET_OF(hkbRigidBodyRagdollControlData_DefaultStruct,m_durationToBlend)},
			1.0f
		};
	}
	hkClass hkbRigidBodyRagdollControlDataClass(
		"hkbRigidBodyRagdollControlData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRigidBodyRagdollControlDataClass_Members),
		HK_COUNT_OF(hkbRigidBodyRagdollControlDataClass_Members),
		&hkbRigidBodyRagdollControlData_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRigidBodyRagdollControlsModifierClass_Members[] =
	{
		{ "controlData", &hkbRigidBodyRagdollControlDataClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbRigidBodyRagdollControlsModifierClass(
		"hkbRigidBodyRagdollControlsModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRigidBodyRagdollControlsModifierClass_Members),
		HK_COUNT_OF(hkbRigidBodyRagdollControlsModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRigidBodyRagdollModifierClass_Members[] =
	{
		{ "controlDataPalette", &hkaKeyFrameHierarchyUtilityControlDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "bodyIndexToPaletteIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "keyframedBonesList", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "keyframedBones", &hkBitFieldClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rigidBodyController", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "doSetupNextEvaluate", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeSinceLastModify", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeSinceBegin", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbRigidBodyRagdollModifier_DefaultStruct
		{
			int s_defaultOffsets[8];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_doSetupNextEvaluate;
		};
		const hkbRigidBodyRagdollModifier_DefaultStruct hkbRigidBodyRagdollModifier_Default =
		{
			{-1,-1,-1,-1,-1,HK_OFFSET_OF(hkbRigidBodyRagdollModifier_DefaultStruct,m_doSetupNextEvaluate),-1,-1},
			true
		};
	}
	hkClass hkbRigidBodyRagdollModifierClass(
		"hkbRigidBodyRagdollModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRigidBodyRagdollModifierClass_Members),
		HK_COUNT_OF(hkbRigidBodyRagdollModifierClass_Members),
		&hkbRigidBodyRagdollModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRotateCharacterModifierClass_Members[] =
	{
		{ "degreesPerSecond", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "axisOfRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbRotateCharacterModifier_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_degreesPerSecond;
			_hkVector4 m_axisOfRotation;
		};
		const hkbRotateCharacterModifier_DefaultStruct hkbRotateCharacterModifier_Default =
		{
			{HK_OFFSET_OF(hkbRotateCharacterModifier_DefaultStruct,m_degreesPerSecond),HK_OFFSET_OF(hkbRotateCharacterModifier_DefaultStruct,m_axisOfRotation),-1},
		1.0f,	{1,0,0}
		};
	}
	hkClass hkbRotateCharacterModifierClass(
		"hkbRotateCharacterModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRotateCharacterModifierClass_Members),
		HK_COUNT_OF(hkbRotateCharacterModifierClass_Members),
		&hkbRotateCharacterModifier_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbSenseHandleModifierSensingModeEnumItems[] =
	{
		{0, "SENSE_IN_NEARBY_RIGID_BODIES"},
		{1, "SENSE_IN_RIGID_BODIES_OUTSIDE_THIS_CHARACTER"},
		{2, "SENSE_IN_OTHER_CHARACTER_RIGID_BODIES"},
		{3, "SENSE_IN_THIS_CHARACTER_RIGID_BODIES"},
		{4, "SENSE_IN_GIVEN_CHARACTER_RIGID_BODIES"},
		{5, "SENSE_IN_GIVEN_RIGID_BODY"},
		{6, "SENSE_IN_OTHER_CHARACTER_SKELETON"},
		{7, "SENSE_IN_THIS_CHARACTER_SKELETON"},
		{8, "SENSE_IN_GIVEN_CHARACTER_SKELETON"},
	};
	static const hkInternalClassEnum hkbSenseHandleModifierEnums[] = {
		{"SensingMode", hkbSenseHandleModifierSensingModeEnumItems, 9, HK_NULL, 0 }
	};
	const hkClassEnum* hkbSenseHandleModifierSensingModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbSenseHandleModifierEnums[0]);
	static hkInternalClassMember hkbSenseHandleModifierClass_Members[] =
	{
		{ "handle", &hkbHandleClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "sensorLocalOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "handleOut", &hkbHandleClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "handleIn", &hkbHandleClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "localFrameName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensorLocalFrameName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "distanceOut", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensorRagdollBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensorAnimationBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensingMode", HK_NULL, hkbSenseHandleModifierSensingModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "extrapolateSensorPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "keepFirstSensedHandle", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "foundHandleOut", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "timeSinceLastModify", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbSenseHandleModifier_DefaultStruct
		{
			int s_defaultOffsets[15];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_sensorLocalOffset;
			hkReal m_maxDistance;
			hkInt16 m_sensorRagdollBoneIndex;
			hkInt16 m_sensorAnimationBoneIndex;
		};
		const hkbSenseHandleModifier_DefaultStruct hkbSenseHandleModifier_Default =
		{
			{-1,HK_OFFSET_OF(hkbSenseHandleModifier_DefaultStruct,m_sensorLocalOffset),-1,-1,-1,-1,HK_OFFSET_OF(hkbSenseHandleModifier_DefaultStruct,m_maxDistance),-1,HK_OFFSET_OF(hkbSenseHandleModifier_DefaultStruct,m_sensorRagdollBoneIndex),HK_OFFSET_OF(hkbSenseHandleModifier_DefaultStruct,m_sensorAnimationBoneIndex),-1,-1,-1,-1,-1},
			{0,0,0},1.0f,-1,-1
		};
	}
	hkClass hkbSenseHandleModifierClass(
		"hkbSenseHandleModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbSenseHandleModifierEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbSenseHandleModifierClass_Members),
		HK_COUNT_OF(hkbSenseHandleModifierClass_Members),
		&hkbSenseHandleModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbTimerModifierClass_Members[] =
	{
		{ "alarmTimeSeconds", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "eventIdToSend", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "secondsElapsed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbTimerModifier_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_eventIdToSend;
		};
		const hkbTimerModifier_DefaultStruct hkbTimerModifier_Default =
		{
			{-1,HK_OFFSET_OF(hkbTimerModifier_DefaultStruct,m_eventIdToSend),-1},
			-1
		};
	}
	hkClass hkbTimerModifierClass(
		"hkbTimerModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbTimerModifierClass_Members),
		HK_COUNT_OF(hkbTimerModifierClass_Members),
		&hkbTimerModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbTransformVectorModifierClass_Members[] =
	{
		{ "rotation", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "translation", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vectorIn", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vectorOut", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotateOnly", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "inverse", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "computeOnActivate", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "computeOnModify", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbTransformVectorModifier_DefaultStruct
		{
			int s_defaultOffsets[8];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkQuaternion m_rotation;
			_hkBool m_computeOnActivate;
		};
		const hkbTransformVectorModifier_DefaultStruct hkbTransformVectorModifier_Default =
		{
			{HK_OFFSET_OF(hkbTransformVectorModifier_DefaultStruct,m_rotation),-1,-1,-1,-1,-1,HK_OFFSET_OF(hkbTransformVectorModifier_DefaultStruct,m_computeOnActivate),-1},
			{0,0,0,1},true
		};
	}
	hkClass hkbTransformVectorModifierClass(
		"hkbTransformVectorModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbTransformVectorModifierClass_Members),
		HK_COUNT_OF(hkbTransformVectorModifierClass_Members),
		&hkbTransformVectorModifier_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbTwistModifierSetAngleMethodEnumItems[] =
	{
		{0, "LINEAR"},
		{1, "RAMPED"},
	};
	static const hkInternalClassEnum hkbTwistModifierEnums[] = {
		{"SetAngleMethod", hkbTwistModifierSetAngleMethodEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbTwistModifierSetAngleMethodEnum = reinterpret_cast<const hkClassEnum*>(&hkbTwistModifierEnums[0]);
	static hkInternalClassMember hkbTwistModifierClass_Members[] =
	{
		{ "setAngleMethod", HK_NULL, hkbTwistModifierSetAngleMethodEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "isAdditive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "endBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "startBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "axisOfRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "twistAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbTwistModifier_DefaultStruct
		{
			int s_defaultOffsets[6];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_isAdditive;
			hkInt16 m_endBoneIndex;
			hkInt16 m_startBoneIndex;
			_hkVector4 m_axisOfRotation;
		};
		const hkbTwistModifier_DefaultStruct hkbTwistModifier_Default =
		{
			{-1,HK_OFFSET_OF(hkbTwistModifier_DefaultStruct,m_isAdditive),HK_OFFSET_OF(hkbTwistModifier_DefaultStruct,m_endBoneIndex),HK_OFFSET_OF(hkbTwistModifier_DefaultStruct,m_startBoneIndex),HK_OFFSET_OF(hkbTwistModifier_DefaultStruct,m_axisOfRotation),-1},
		1,-1,-1,	{1,0,0}
		};
	}
	hkClass hkbTwistModifierClass(
		"hkbTwistModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbTwistModifierEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbTwistModifierClass_Members),
		HK_COUNT_OF(hkbTwistModifierClass_Members),
		&hkbTwistModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbModifierWrapperClass_Members[] =
	{
		{ "modifier", &hkbModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbModifierWrapperClass(
		"hkbModifierWrapper",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbModifierWrapperClass_Members),
		HK_COUNT_OF(hkbModifierWrapperClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbNodeGetChildrenFlagBitsEnumItems[] =
	{
		{1/*0x1*/, "FLAG_ACTIVE_ONLY"},
		{2/*0x2*/, "FLAG_GENERATORS_ONLY"},
	};
	static const hkInternalClassEnum hkbNodeEnums[] = {
		{"GetChildrenFlagBits", hkbNodeGetChildrenFlagBitsEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbNodeGetChildrenFlagBitsEnum = reinterpret_cast<const hkClassEnum*>(&hkbNodeEnums[0]);
	static hkInternalClassMember hkbNodeClass_Members[] =
	{
		{ "userData", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "variableBindingSet", &hkbVariableBindingSetClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbNodeClass(
		"hkbNode",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbNodeEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbNodeClass_Members),
		HK_COUNT_OF(hkbNodeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkbPredicateClass(
		"hkbPredicate",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbStringPredicateClass_Members[] =
	{
		{ "predicateString", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbStringPredicateClass(
		"hkbStringPredicate",
		&hkbPredicateClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbStringPredicateClass_Members),
		HK_COUNT_OF(hkbStringPredicateClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbProjectDataClass_Members[] =
	{
		{ "attachmentSetups", &hkbAttachmentSetupClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "worldUpWS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stringData", &hkbProjectStringDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbProjectDataClass(
		"hkbProjectData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbProjectDataClass_Members),
		HK_COUNT_OF(hkbProjectDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbProjectStringDataClass_Members[] =
	{
		{ "animationPath", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "behaviorPath", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "characterPath", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "animationFilenames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "behaviorFilenames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "characterFilenames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "eventNames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL }
	};
	hkClass hkbProjectStringDataClass(
		"hkbProjectStringData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbProjectStringDataClass_Members),
		HK_COUNT_OF(hkbProjectStringDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbSequenceClass_Members[] =
	{
		{ "eventSequencedData", &hkbEventSequencedDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "realVariableSequencedData", &hkbRealVariableSequencedDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "boolVariableSequencedData", &hkbBoolVariableSequencedDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "intVariableSequencedData", &hkbIntVariableSequencedDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "enableEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "disableEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stringData", &hkbSequenceStringDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "variableIdMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "eventIdMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbSequence_DefaultStruct
		{
			int s_defaultOffsets[11];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_enableEventId;
			hkInt32 m_disableEventId;
		};
		const hkbSequence_DefaultStruct hkbSequence_Default =
		{
			{-1,-1,-1,-1,HK_OFFSET_OF(hkbSequence_DefaultStruct,m_enableEventId),HK_OFFSET_OF(hkbSequence_DefaultStruct,m_disableEventId),-1,-1,-1,-1,-1},
			-1,-1
		};
	}
	hkClass hkbSequenceClass(
		"hkbSequence",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbSequenceClass_Members),
		HK_COUNT_OF(hkbSequenceClass_Members),
		&hkbSequence_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbSequenceStringDataClass_Members[] =
	{
		{ "eventNames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "variableNames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL }
	};
	hkClass hkbSequenceStringDataClass(
		"hkbSequenceStringData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbSequenceStringDataClass_Members),
		HK_COUNT_OF(hkbSequenceStringDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkbSequencedDataClass(
		"hkbSequencedData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbEventSequencedData_SequencedEventClass_Members[] =
	{
		{ "event", &hkbEventClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbEventSequencedDataSequencedEventClass(
		"hkbEventSequencedDataSequencedEvent",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbEventSequencedData_SequencedEventClass_Members),
		HK_COUNT_OF(hkbEventSequencedData_SequencedEventClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbEventSequencedDataClass_Members[] =
	{
		{ "events", &hkbEventSequencedDataSequencedEventClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "nextEvent", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbEventSequencedDataClass(
		"hkbEventSequencedData",
		&hkbSequencedDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbEventSequencedDataClass_Members),
		HK_COUNT_OF(hkbEventSequencedDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRealVariableSequencedData_SampleClass_Members[] =
	{
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbRealVariableSequencedDataSampleClass(
		"hkbRealVariableSequencedDataSample",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRealVariableSequencedData_SampleClass_Members),
		HK_COUNT_OF(hkbRealVariableSequencedData_SampleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRealVariableSequencedDataClass_Members[] =
	{
		{ "samples", &hkbRealVariableSequencedDataSampleClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "variableIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "nextSample", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbRealVariableSequencedData_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_variableIndex;
		};
		const hkbRealVariableSequencedData_DefaultStruct hkbRealVariableSequencedData_Default =
		{
			{-1,HK_OFFSET_OF(hkbRealVariableSequencedData_DefaultStruct,m_variableIndex),-1},
			-1
		};
	}
	hkClass hkbRealVariableSequencedDataClass(
		"hkbRealVariableSequencedData",
		&hkbSequencedDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRealVariableSequencedDataClass_Members),
		HK_COUNT_OF(hkbRealVariableSequencedDataClass_Members),
		&hkbRealVariableSequencedData_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbBoolVariableSequencedData_SampleClass_Members[] =
	{
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbBoolVariableSequencedDataSampleClass(
		"hkbBoolVariableSequencedDataSample",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbBoolVariableSequencedData_SampleClass_Members),
		HK_COUNT_OF(hkbBoolVariableSequencedData_SampleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbBoolVariableSequencedDataClass_Members[] =
	{
		{ "samples", &hkbBoolVariableSequencedDataSampleClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "variableIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "nextSample", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbBoolVariableSequencedData_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_variableIndex;
		};
		const hkbBoolVariableSequencedData_DefaultStruct hkbBoolVariableSequencedData_Default =
		{
			{-1,HK_OFFSET_OF(hkbBoolVariableSequencedData_DefaultStruct,m_variableIndex),-1},
			-1
		};
	}
	hkClass hkbBoolVariableSequencedDataClass(
		"hkbBoolVariableSequencedData",
		&hkbSequencedDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbBoolVariableSequencedDataClass_Members),
		HK_COUNT_OF(hkbBoolVariableSequencedDataClass_Members),
		&hkbBoolVariableSequencedData_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbIntVariableSequencedData_SampleClass_Members[] =
	{
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbIntVariableSequencedDataSampleClass(
		"hkbIntVariableSequencedDataSample",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbIntVariableSequencedData_SampleClass_Members),
		HK_COUNT_OF(hkbIntVariableSequencedData_SampleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbIntVariableSequencedDataClass_Members[] =
	{
		{ "samples", &hkbIntVariableSequencedDataSampleClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "variableIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "nextSample", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbIntVariableSequencedData_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_variableIndex;
		};
		const hkbIntVariableSequencedData_DefaultStruct hkbIntVariableSequencedData_Default =
		{
			{-1,HK_OFFSET_OF(hkbIntVariableSequencedData_DefaultStruct,m_variableIndex),-1},
			-1
		};
	}
	hkClass hkbIntVariableSequencedDataClass(
		"hkbIntVariableSequencedData",
		&hkbSequencedDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbIntVariableSequencedDataClass_Members),
		HK_COUNT_OF(hkbIntVariableSequencedDataClass_Members),
		&hkbIntVariableSequencedData_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbStateMachineStartStateModeEnumItems[] =
	{
		{0, "START_STATE_MODE_SET_ON_ACTIVATE"},
		{1, "START_STATE_MODE_SET_ONCE"},
		{2, "START_STATE_MODE_SYNC"},
		{3, "START_STATE_MODE_RANDOM"},
	};
	static const hkInternalClassEnumItem hkbStateMachineStateMachineSelfTransitionModeEnumItems[] =
	{
		{0, "SELF_TRANSITION_MODE_NO_TRANSITION"},
		{1, "SELF_TRANSITION_MODE_TRANSITION_TO_START_STATE"},
		{2, "SELF_TRANSITION_MODE_FORCE_TRANSITION_TO_START_STATE"},
	};
	static const hkInternalClassEnum hkbStateMachineEnums[] = {
		{"StartStateMode", hkbStateMachineStartStateModeEnumItems, 4, HK_NULL, 0 },
		{"StateMachineSelfTransitionMode", hkbStateMachineStateMachineSelfTransitionModeEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkbStateMachineStartStateModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbStateMachineEnums[0]);
	const hkClassEnum* hkbStateMachineStateMachineSelfTransitionModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbStateMachineEnums[1]);
	static hkInternalClassMember hkbStateMachine_TimeIntervalClass_Members[] =
	{
		{ "enterEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "exitEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "enterTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "exitTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbStateMachineTimeIntervalClass(
		"hkbStateMachineTimeInterval",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbStateMachine_TimeIntervalClass_Members),
		HK_COUNT_OF(hkbStateMachine_TimeIntervalClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbStateMachineTransitionInfoTransitionFlagsEnumItems[] =
	{
		{1/*0x1*/, "FLAG_USE_TRIGGER_INTERVAL"},
		{2/*0x2*/, "FLAG_USE_INITIATE_INTERVAL"},
		{4/*0x4*/, "FLAG_UNINTERRUPTIBLE_WHILE_PLAYING"},
		{8/*0x8*/, "FLAG_UNINTERRUPTIBLE_WHILE_DELAYED"},
		{16/*0x10*/, "FLAG_DELAY_STATE_CHANGE"},
		{32/*0x20*/, "FLAG_DISABLED"},
		{64/*0x40*/, "FLAG_DISALLOW_RETURN_TO_PREVIOUS_STATE"},
		{128/*0x80*/, "FLAG_DISALLOW_RANDOM_TRANSITION"},
		{256/*0x100*/, "FLAG_DISABLE_PREDICATE"},
		{512/*0x200*/, "FLAG_ALLOW_SELF_TRANSITION_BY_TRANSITION_FROM_ANY_STATE"},
		{1024/*0x400*/, "FLAG_IS_GLOBAL_WILDCARD"},
		{2048/*0x800*/, "FLAG_IS_LOCAL_WILDCARD"},
		{4096/*0x1000*/, "FLAG_FROM_NESTED_STATE_ID_IS_VALID"},
		{8192/*0x2000*/, "FLAG_TO_NESTED_STATE_ID_IS_VALID"},
		{16384/*0x4000*/, "FLAG_ABUT_AT_END_OF_FROM_GENERATOR"},
	};
	static const hkInternalClassEnumItem hkbStateMachineTransitionInfoInternalFlagBitsEnumItems[] =
	{
		{1/*0x1*/, "FLAG_INTERNAL_IN_TRIGGER_INTERVAL"},
		{2/*0x2*/, "FLAG_INTERNAL_IN_INITIATE_INTERVAL"},
	};
	static const hkInternalClassEnum hkbStateMachineTransitionInfoEnums[] = {
		{"TransitionFlags", hkbStateMachineTransitionInfoTransitionFlagsEnumItems, 15, HK_NULL, 0 },
		{"InternalFlagBits", hkbStateMachineTransitionInfoInternalFlagBitsEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbStateMachineTransitionInfoTransitionFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkbStateMachineTransitionInfoEnums[0]);
	const hkClassEnum* hkbStateMachineTransitionInfoInternalFlagBitsEnum = reinterpret_cast<const hkClassEnum*>(&hkbStateMachineTransitionInfoEnums[1]);
	static hkInternalClassMember hkbStateMachine_TransitionInfoClass_Members[] =
	{
		{ "triggerInterval", &hkbStateMachineTimeIntervalClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initiateInterval", &hkbStateMachineTimeIntervalClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transition", &hkbTransitionEffectClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "predicate", &hkbPredicateClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "eventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "toStateId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fromNestedStateId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "toNestedStateId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "priority", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "flags", HK_NULL, hkbStateMachineTransitionInfoTransitionFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "internalFlags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbStateMachineTransitionInfoClass(
		"hkbStateMachineTransitionInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbStateMachineTransitionInfoEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbStateMachine_TransitionInfoClass_Members),
		HK_COUNT_OF(hkbStateMachine_TransitionInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbStateMachine_ActiveTransitionInfoClass_Members[] =
	{
		{ "transitionInfo", &hkbStateMachineTransitionInfoClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "transitionEffect", &hkbTransitionEffectClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "fromStateId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "toStateId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isReturnToPreviousState", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbStateMachineActiveTransitionInfoClass(
		"hkbStateMachineActiveTransitionInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbStateMachine_ActiveTransitionInfoClass_Members),
		HK_COUNT_OF(hkbStateMachine_ActiveTransitionInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbStateMachine_ProspectiveTransitionInfoClass_Members[] =
	{
		{ "transitionInfo", &hkbStateMachineTransitionInfoClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "transitionEffect", &hkbTransitionEffectClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "toStateId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isGlobalWildcard", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbStateMachineProspectiveTransitionInfoClass(
		"hkbStateMachineProspectiveTransitionInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbStateMachine_ProspectiveTransitionInfoClass_Members),
		HK_COUNT_OF(hkbStateMachine_ProspectiveTransitionInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbStateMachine_StateInfoClass_Members[] =
	{
		{ "enterNotifyEvent", &hkbEventClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "exitNotifyEvent", &hkbEventClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "generator", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transitions", &hkbStateMachineTransitionInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "stateId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "inPackfile", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbStateMachineStateInfoClass(
		"hkbStateMachineStateInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbStateMachine_StateInfoClass_Members),
		HK_COUNT_OF(hkbStateMachine_StateInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbStateMachine_NestedStateMachineDataClass_Members[] =
	{
		{ "nestedStateMachine", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "eventIdMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbStateMachineNestedStateMachineDataClass(
		"hkbStateMachineNestedStateMachineData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbStateMachine_NestedStateMachineDataClass_Members),
		HK_COUNT_OF(hkbStateMachine_NestedStateMachineDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbStateMachineClass_Members[] =
	{
		{ "eventToSendWhenStateOrTransitionChanges", &hkbEventClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "startStateId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "returnToPreviousStateEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "randomTransitionEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transitionToNextHigherStateEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transitionToNextLowerStateEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "syncVariableIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wrapAroundStateId", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxSimultaneousTransitions", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "startStateMode", HK_NULL, hkbStateMachineStartStateModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "enableGlobalTransitions", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "enableNestedTransitions", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "selfTransitionMode", HK_NULL, hkbStateMachineStateMachineSelfTransitionModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "states", &hkbStateMachineStateInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "globalTransitions", &hkbStateMachineTransitionInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "activeTransitions", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "delayedTransition", &hkbStateMachineProspectiveTransitionInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "stateIdToIndexMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "eventQueue", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "nestedStateMachineData", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeInState", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "currentStateId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "previousStateId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "nextStartStateIndexOverride", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isActive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "stateOrTransitionChanged", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isDelayedTransitionReturnToPreviousState", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "resetLocalTime", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "wasInAbutRangeLastFrame", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "echoNextUpdate", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbStateMachine_DefaultStruct
		{
			int s_defaultOffsets[30];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_transitionToNextHigherStateEventId;
			hkInt32 m_transitionToNextLowerStateEventId;
			hkInt32 m_syncVariableIndex;
			_hkBool m_wrapAroundStateId;
			hkInt8 m_maxSimultaneousTransitions;
		};
		const hkbStateMachine_DefaultStruct hkbStateMachine_Default =
		{
			{-1,-1,-1,-1,HK_OFFSET_OF(hkbStateMachine_DefaultStruct,m_transitionToNextHigherStateEventId),HK_OFFSET_OF(hkbStateMachine_DefaultStruct,m_transitionToNextLowerStateEventId),HK_OFFSET_OF(hkbStateMachine_DefaultStruct,m_syncVariableIndex),HK_OFFSET_OF(hkbStateMachine_DefaultStruct,m_wrapAroundStateId),HK_OFFSET_OF(hkbStateMachine_DefaultStruct,m_maxSimultaneousTransitions),-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
			-1,-1,-1,true,32
		};
	}
	hkClass hkbStateMachineClass(
		"hkbStateMachine",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbStateMachineEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbStateMachineClass_Members),
		HK_COUNT_OF(hkbStateMachineClass_Members),
		&hkbStateMachine_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbTransitionEffectSelfTransitionModeEnumItems[] =
	{
		{0, "SELF_TRANSITION_MODE_CONTINUE_IF_CYCLIC_BLEND_IF_ACYCLIC"},
		{1, "SELF_TRANSITION_MODE_CONTINUE"},
		{2, "SELF_TRANSITION_MODE_RESET"},
		{3, "SELF_TRANSITION_MODE_BLEND"},
	};
	static const hkInternalClassEnumItem hkbTransitionEffectEventModeEnumItems[] =
	{
		{0, "EVENT_MODE_DEFAULT"},
		{1, "EVENT_MODE_PROCESS_ALL"},
		{2, "EVENT_MODE_IGNORE_FROM_GENERATOR"},
	};
	static const hkInternalClassEnum hkbTransitionEffectEnums[] = {
		{"SelfTransitionMode", hkbTransitionEffectSelfTransitionModeEnumItems, 4, HK_NULL, 0 },
		{"EventMode", hkbTransitionEffectEventModeEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkbTransitionEffectSelfTransitionModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbTransitionEffectEnums[0]);
	const hkClassEnum* hkbTransitionEffectEventModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbTransitionEffectEnums[1]);
	static hkInternalClassMember hkbTransitionEffectClass_Members[] =
	{
		{ "selfTransitionMode", HK_NULL, hkbTransitionEffectSelfTransitionModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "eventMode", HK_NULL, hkbTransitionEffectEventModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkbTransitionEffectClass(
		"hkbTransitionEffect",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbTransitionEffectEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbTransitionEffectClass_Members),
		HK_COUNT_OF(hkbTransitionEffectClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbBlendingTransitionEffectFlagBitsEnumItems[] =
	{
		{0/*0x0*/, "FLAG_NONE"},
		{1/*0x1*/, "FLAG_IGNORE_FROM_WORLD_FROM_MODEL"},
		{2/*0x2*/, "FLAG_SYNC"},
	};
	static const hkInternalClassEnumItem hkbBlendingTransitionEffectEndModeEnumItems[] =
	{
		{0, "END_MODE_NONE"},
		{1, "END_MODE_TRANSITION_UNTIL_END_OF_FROM_GENERATOR"},
		{2, "END_MODE_CAP_DURATION_AT_END_OF_FROM_GENERATOR"},
	};
	static const hkInternalClassEnum hkbBlendingTransitionEffectEnums[] = {
		{"FlagBits", hkbBlendingTransitionEffectFlagBitsEnumItems, 3, HK_NULL, 0 },
		{"EndMode", hkbBlendingTransitionEffectEndModeEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkbBlendingTransitionEffectFlagBitsEnum = reinterpret_cast<const hkClassEnum*>(&hkbBlendingTransitionEffectEnums[0]);
	const hkClassEnum* hkbBlendingTransitionEffectEndModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbBlendingTransitionEffectEnums[1]);
	static hkInternalClassMember hkbBlendingTransitionEffectClass_Members[] =
	{
		{ "duration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "toGeneratorStartTimeFraction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "flags", HK_NULL, hkbBlendingTransitionEffectFlagBitsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "endMode", HK_NULL, hkbBlendingTransitionEffectEndModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "blendCurve", HK_NULL, hkbBlendCurveUtilsBlendCurveEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "fromGenerator", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "toGenerator", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeRemaining", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeInTransition", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isClone", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "applySelfTransition", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbBlendingTransitionEffect_DefaultStruct
		{
			int s_defaultOffsets[11];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint16 /* hkFlags< FlagBits, hkUint16 > */ m_flags;
		};
		const hkbBlendingTransitionEffect_DefaultStruct hkbBlendingTransitionEffect_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkbBlendingTransitionEffect_DefaultStruct,m_flags),-1,-1,-1,-1,-1,-1,-1,-1},
			0x0
		};
	}
	hkClass hkbBlendingTransitionEffectClass(
		"hkbBlendingTransitionEffect",
		&hkbTransitionEffectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbBlendingTransitionEffectEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbBlendingTransitionEffectClass_Members),
		HK_COUNT_OF(hkbBlendingTransitionEffectClass_Members),
		&hkbBlendingTransitionEffect_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbExtrapolatingTransitionEffectClass_Members[] =
	{
		{ "fromGeneratorOutput", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "fromGeneratorSyncInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "additivePose", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "motion", HK_NULL, HK_NULL, hkClassMember::TYPE_QSTRANSFORM, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "toGeneratorDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isFromGeneratorActive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbExtrapolatingTransitionEffectClass(
		"hkbExtrapolatingTransitionEffect",
		&hkbBlendingTransitionEffectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbExtrapolatingTransitionEffectClass_Members),
		HK_COUNT_OF(hkbExtrapolatingTransitionEffectClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbGeneratorTransitionEffectToGeneratorStateEnumItems[] =
	{
		{0, "STATE_INACTIVE"},
		{1, "STATE_READY_FOR_SET_LOCAL_TIME"},
		{2, "STATE_READY_FOR_APPLY_SELF_TRANSITION_MODE"},
		{3, "STATE_ACTIVE"},
	};
	static const hkInternalClassEnum hkbGeneratorTransitionEffectEnums[] = {
		{"ToGeneratorState", hkbGeneratorTransitionEffectToGeneratorStateEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkbGeneratorTransitionEffectToGeneratorStateEnum = reinterpret_cast<const hkClassEnum*>(&hkbGeneratorTransitionEffectEnums[0]);
	static hkInternalClassMember hkbGeneratorTransitionEffectClass_Members[] =
	{
		{ "transitionGenerator", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "fromGenerator", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "toGenerator", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeInTransition", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "duration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "blendInDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blendOutDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isClone", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "toGeneratorState", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "echoTransitionGenerator", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "justActivated", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbGeneratorTransitionEffectClass(
		"hkbGeneratorTransitionEffect",
		&hkbTransitionEffectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbGeneratorTransitionEffectEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbGeneratorTransitionEffectClass_Members),
		HK_COUNT_OF(hkbGeneratorTransitionEffectClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbBlendCurveUtilsBlendCurveEnumItems[] =
	{
		{0, "BLEND_CURVE_SMOOTH"},
		{1, "BLEND_CURVE_LINEAR"},
		{2, "BLEND_CURVE_LINEAR_TO_SMOOTH"},
		{3, "BLEND_CURVE_SMOOTH_TO_LINEAR"},
	};
	static const hkInternalClassEnum hkbBlendCurveUtilsEnums[] = {
		{"BlendCurve", hkbBlendCurveUtilsBlendCurveEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkbBlendCurveUtilsBlendCurveEnum = reinterpret_cast<const hkClassEnum*>(&hkbBlendCurveUtilsEnums[0]);
	hkClass hkbBlendCurveUtilsClass(
		"hkbBlendCurveUtils",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbBlendCurveUtilsEnums),
		1,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbHandleClass_Members[] =
	{
		{ "frame", &hkLocalFrameClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "rigidBody", &hkpRigidBodyClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "character", &hkbCharacterClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "animationBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbHandleClass(
		"hkbHandle",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbHandleClass_Members),
		HK_COUNT_OF(hkbHandleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbVariableBindingSetBindingInternalBindingFlagsEnumItems[] =
	{
		{1, "FLAG_OUTPUT"},
	};
	static const hkInternalClassEnum hkbVariableBindingSetBindingEnums[] = {
		{"InternalBindingFlags", hkbVariableBindingSetBindingInternalBindingFlagsEnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hkbVariableBindingSetBindingInternalBindingFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkbVariableBindingSetBindingEnums[0]);
	static hkInternalClassMember hkbVariableBindingSet_BindingClass_Members[] =
	{
		{ "object", &hkReferencedObjectClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "memberPath", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "memberDataPtr", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "memberClass", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "variableIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bitIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "memberType", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "flags", HK_NULL, HK_NULL, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_INT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbVariableBindingSetBinding_DefaultStruct
		{
			int s_defaultOffsets[8];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt8 m_bitIndex;
		};
		const hkbVariableBindingSetBinding_DefaultStruct hkbVariableBindingSetBinding_Default =
		{
			{-1,-1,-1,-1,-1,HK_OFFSET_OF(hkbVariableBindingSetBinding_DefaultStruct,m_bitIndex),-1,-1},
			-1
		};
	}
	hkClass hkbVariableBindingSetBindingClass(
		"hkbVariableBindingSetBinding",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbVariableBindingSetBindingEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbVariableBindingSet_BindingClass_Members),
		HK_COUNT_OF(hkbVariableBindingSet_BindingClass_Members),
		&hkbVariableBindingSetBinding_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbVariableBindingSetClass_Members[] =
	{
		{ "bindings", &hkbVariableBindingSetBindingClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbVariableBindingSetClass(
		"hkbVariableBindingSet",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbVariableBindingSetClass_Members),
		HK_COUNT_OF(hkbVariableBindingSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbVariableInfoVariableTypeEnumItems[] =
	{
		{-1, "VARIABLE_TYPE_INVALID"},
		{0, "VARIABLE_TYPE_BOOL"},
		{1, "VARIABLE_TYPE_INT8"},
		{2, "VARIABLE_TYPE_INT16"},
		{3, "VARIABLE_TYPE_INT32"},
		{4, "VARIABLE_TYPE_REAL"},
		{5, "VARIABLE_TYPE_POINTER"},
		{6, "VARIABLE_TYPE_VECTOR3"},
		{7, "VARIABLE_TYPE_VECTOR4"},
		{8, "VARIABLE_TYPE_QUATERNION"},
	};
	static const hkInternalClassEnum hkbVariableInfoEnums[] = {
		{"VariableType", hkbVariableInfoVariableTypeEnumItems, 10, HK_NULL, 0 }
	};
	const hkClassEnum* hkbVariableInfoVariableTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkbVariableInfoEnums[0]);
	static hkInternalClassMember hkbVariableInfoClass_Members[] =
	{
		{ "initialValue", &hkbVariableValueClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, hkbVariableInfoVariableTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkbVariableInfoClass(
		"hkbVariableInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbVariableInfoEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbVariableInfoClass_Members),
		HK_COUNT_OF(hkbVariableInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbVariableValueClass_Members[] =
	{
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbVariableValueClass(
		"hkbVariableValue",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbVariableValueClass_Members),
		HK_COUNT_OF(hkbVariableValueClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbSimpleCharacterClass_Members[] =
	{
		{ "nearbyCharacters", &hkbCharacterClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "eventQueue", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "ragdollInstance", &hkaRagdollInstanceClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "setup", &hkbCharacterSetupClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "worldFromModel", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_QSTRANSFORM, 0, 0, 0, HK_NULL },
		{ "poseLocal", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_QSTRANSFORM, 0, 0, 0, HK_NULL },
		{ "deleteWorldFromModel", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deletePoseLocal", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbSimpleCharacterClass(
		"hkbSimpleCharacter",
		&hkbCharacterClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbSimpleCharacterClass_Members),
		HK_COUNT_OF(hkbSimpleCharacterClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbDemoConfigCharacterInfoClass_Members[] =
	{
		{ "rigFilename", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "skinFilenames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "behaviorFilename", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "characterDataFilename", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "attachmentsFilename", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initialPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initialRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "modelUpAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollBoneLayers", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL }
	};
	hkClass hkbDemoConfigCharacterInfoClass(
		"hkbDemoConfigCharacterInfo",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbDemoConfigCharacterInfoClass_Members),
		HK_COUNT_OF(hkbDemoConfigCharacterInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbDemoConfig_TerrainInfoClass_Members[] =
	{
		{ "filename", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "layer", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "systemGroup", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "createDisplayObjects", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "terrainRigidBody", &hkpRigidBodyClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbDemoConfigTerrainInfoClass(
		"hkbDemoConfigTerrainInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbDemoConfig_TerrainInfoClass_Members),
		HK_COUNT_OF(hkbDemoConfig_TerrainInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbDemoConfig_StickVariableInfoClass_Members[] =
	{
		{ "variableName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minValue", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxValue", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbDemoConfigStickVariableInfoClass(
		"hkbDemoConfigStickVariableInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbDemoConfig_StickVariableInfoClass_Members),
		HK_COUNT_OF(hkbDemoConfig_StickVariableInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbDemoConfigClass_Members[] =
	{
		{ "characterInfo", &hkbDemoConfigCharacterInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "terrainInfo", &hkbDemoConfigTerrainInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "skinAttributeIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "buttonPressToEventMap", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 16, 0, 0, HK_NULL },
		{ "buttonReleaseToEventMap", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 16, 0, 0, HK_NULL },
		{ "worldUpAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extraCharacterClones", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numTracks", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "proxyHeight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "proxyRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "proxyOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rootPath", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "projectDataFilename", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useAttachments", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useProxy", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useSkyBox", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useTrackingCamera", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "accumulateMotion", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "testCloning", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useSplineCompression", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stickVariables", &hkbDemoConfigStickVariableInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 4, 0, 0, HK_NULL },
		{ "gamePadToRotateTerrainAboutItsAxisMap", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 6, 0, 0, HK_NULL },
		{ "gamePadToAddRemoveCharacterMap", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "filter", &hkpGroupFilterClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbDemoConfig_DefaultStruct
		{
			int s_defaultOffsets[24];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_numTracks;
			_hkBool m_testCloning;
		};
		const hkbDemoConfig_DefaultStruct hkbDemoConfig_Default =
		{
			{-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkbDemoConfig_DefaultStruct,m_numTracks),-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkbDemoConfig_DefaultStruct,m_testCloning),-1,-1,-1,-1,-1},
			14,true
		};
	}
	hkClass hkbDemoConfigClass(
		"hkbDemoConfig",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbDemoConfigClass_Members),
		HK_COUNT_OF(hkbDemoConfigClass_Members),
		&hkbDemoConfig_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbAnimatedSkeletonGeneratorClass_Members[] =
	{
		{ "animatedSkeleton", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbAnimatedSkeletonGeneratorClass(
		"hkbAnimatedSkeletonGenerator",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbAnimatedSkeletonGeneratorClass_Members),
		HK_COUNT_OF(hkbAnimatedSkeletonGeneratorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbCustomTestGeneratorModesEnumItems[] =
	{
		{0, "MODE_ALA"},
		{1, "MODE_DEPECHE"},
		{5, "MODE_FURIOUS"},
	};
	static const hkInternalClassEnumItem hkbCustomTestGeneratorStrangeFlagsEnumItems[] =
	{
		{1/*0x1*/, "FLAG_UNO"},
		{2/*0x2*/, "FLAG_ZWEI"},
		{4/*0x4*/, "FLAG_SHI_OR_YON"},
		{240/*0xf0*/, "FLAG_LOTS_O_BITS"},
	};
	static const hkInternalClassEnum hkbCustomTestGeneratorEnums[] = {
		{"Modes", hkbCustomTestGeneratorModesEnumItems, 3, HK_NULL, 0 },
		{"StrangeFlags", hkbCustomTestGeneratorStrangeFlagsEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkbCustomTestGeneratorModesEnum = reinterpret_cast<const hkClassEnum*>(&hkbCustomTestGeneratorEnums[0]);
	const hkClassEnum* hkbCustomTestGeneratorStrangeFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkbCustomTestGeneratorEnums[1]);
	static hkInternalClassMember hkbCustomTestGenerator_StruckClass_Members[] =
	{
		{ "hkBool", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "string", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "int", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkInt8", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkInt16", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkInt32", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkUint8", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkUint16", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkUint32", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkReal", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "mode_hkInt8", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "mode_hkInt16", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "mode_hkInt32", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "mode_hkUint8", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "mode_hkUint16", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "mode_hkUint32", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "flags_hkInt8", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "flags_hkInt16", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "flags_hkInt32", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "flags_hkUint8", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "flags_hkUint16", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "flags_hkUint32", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "generator1", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "generator2", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "modifier1", &hkbModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "modifier2", &hkbModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbCustomTestGeneratorStruckClass(
		"hkbCustomTestGeneratorStruck",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbCustomTestGenerator_StruckClass_Members),
		HK_COUNT_OF(hkbCustomTestGenerator_StruckClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbCustomTestGeneratorClass_Members[] =
	{
		{ "hkBool", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "string", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "int", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkInt8", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkInt16", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkInt32", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkUint8", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkUint16", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkUint32", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkReal", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkVector4", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkQuaternion", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkRigidBody", &hkpRigidBodyClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "mode_hkInt8", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "mode_hkInt16", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "mode_hkInt32", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "mode_hkUint8", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "mode_hkUint16", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "mode_hkUint32", HK_NULL, hkbCustomTestGeneratorModesEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "flags_hkInt8", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "flags_hkInt16", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "flags_hkInt32", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "flags_hkUint8", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "flags_hkUint16", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "flags_hkUint32", HK_NULL, hkbCustomTestGeneratorStrangeFlagsEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "myInt", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "generator1", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "generator2", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "modifier1", &hkbModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "modifier2", &hkbModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "array_hkBool", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_BOOL, 0, 0, 0, HK_NULL },
		{ "array_int", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "array_hkInt8", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "array_hkInt16", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "array_hkInt32", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "array_hkUint8", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "array_hkUint16", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "array_hkUint32", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "array_hkReal", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "array_hkbGenerator", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "array_hkbModifier", &hkbModifierClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "Struck", &hkbCustomTestGeneratorStruckClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "array_Struck", &hkbCustomTestGeneratorStruckClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbCustomTestGeneratorClass(
		"hkbCustomTestGenerator",
		&hkbReferencePoseGeneratorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbCustomTestGeneratorEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbCustomTestGeneratorClass_Members),
		HK_COUNT_OF(hkbCustomTestGeneratorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbPoseStoringGeneratorOutputListener_StoredPoseClass_Members[] =
	{
		{ "node", &hkbNodeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "pose", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_QSTRANSFORM, 0, 0, 0, HK_NULL },
		{ "worldFromModel", HK_NULL, HK_NULL, hkClassMember::TYPE_QSTRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isPoseValid", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbPoseStoringGeneratorOutputListenerStoredPoseClass(
		"hkbPoseStoringGeneratorOutputListenerStoredPose",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbPoseStoringGeneratorOutputListener_StoredPoseClass_Members),
		HK_COUNT_OF(hkbPoseStoringGeneratorOutputListener_StoredPoseClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbPoseStoringGeneratorOutputListenerClass_Members[] =
	{
		{ "storedPoses", &hkbPoseStoringGeneratorOutputListenerStoredPoseClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "dirty", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "nodeToIndexMap", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbPoseStoringGeneratorOutputListenerClass(
		"hkbPoseStoringGeneratorOutputListener",
		&hkbGeneratorOutputListenerClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbPoseStoringGeneratorOutputListenerClass_Members),
		HK_COUNT_OF(hkbPoseStoringGeneratorOutputListenerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRadialSelectorGenerator_GeneratorInfoClass_Members[] =
	{
		{ "generator", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "angle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "radialSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbRadialSelectorGeneratorGeneratorInfoClass(
		"hkbRadialSelectorGeneratorGeneratorInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRadialSelectorGenerator_GeneratorInfoClass_Members),
		HK_COUNT_OF(hkbRadialSelectorGenerator_GeneratorInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRadialSelectorGenerator_GeneratorPairClass_Members[] =
	{
		{ "generators", &hkbRadialSelectorGeneratorGeneratorInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "minAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbRadialSelectorGeneratorGeneratorPairClass(
		"hkbRadialSelectorGeneratorGeneratorPair",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRadialSelectorGenerator_GeneratorPairClass_Members),
		HK_COUNT_OF(hkbRadialSelectorGenerator_GeneratorPairClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbRadialSelectorGeneratorClass_Members[] =
	{
		{ "generatorPairs", &hkbRadialSelectorGeneratorGeneratorPairClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "angle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentGeneratorPairIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "currentEndpointIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "currentFraction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "hasSetLocalTime", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbRadialSelectorGenerator_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_currentGeneratorPairIndex;
			int m_currentEndpointIndex;
		};
		const hkbRadialSelectorGenerator_DefaultStruct hkbRadialSelectorGenerator_Default =
		{
			{-1,-1,-1,HK_OFFSET_OF(hkbRadialSelectorGenerator_DefaultStruct,m_currentGeneratorPairIndex),HK_OFFSET_OF(hkbRadialSelectorGenerator_DefaultStruct,m_currentEndpointIndex),-1,-1},
			-1,-1
		};
	}
	hkClass hkbRadialSelectorGeneratorClass(
		"hkbRadialSelectorGenerator",
		&hkbGeneratorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbRadialSelectorGeneratorClass_Members),
		HK_COUNT_OF(hkbRadialSelectorGeneratorClass_Members),
		&hkbRadialSelectorGenerator_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbBalanceRadialSelectorGeneratorClass_Members[] =
	{
		{ "xAxisMS", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "yAxisMS", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "checkBalanceModifier", &hkbCheckBalanceModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbBalanceRadialSelectorGenerator_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_yAxisMS;
		};
		const hkbBalanceRadialSelectorGenerator_DefaultStruct hkbBalanceRadialSelectorGenerator_Default =
		{
			{-1,HK_OFFSET_OF(hkbBalanceRadialSelectorGenerator_DefaultStruct,m_yAxisMS),-1},
			1
		};
	}
	hkClass hkbBalanceRadialSelectorGeneratorClass(
		"hkbBalanceRadialSelectorGenerator",
		&hkbRadialSelectorGeneratorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbBalanceRadialSelectorGeneratorClass_Members),
		HK_COUNT_OF(hkbBalanceRadialSelectorGeneratorClass_Members),
		&hkbBalanceRadialSelectorGenerator_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbAlignBoneModifierAlignModeABAMEnumItems[] =
	{
		{0, "ALIGN_MODE_CHARACTER_WORLD_FROM_MODEL"},
		{1, "ALIGN_MODE_ANIMATION_SKELETON_BONE"},
	};
	static const hkInternalClassEnumItem hkbAlignBoneModifierAlignTargetModeEnumItems[] =
	{
		{0, "ALIGN_TARGET_MODE_CHARACTER_WORLD_FROM_MODEL"},
		{1, "ALIGN_TARGET_MODE_RAGDOLL_SKELETON_BONE"},
		{2, "ALIGN_TARGET_MODE_ANIMATION_SKELETON_BONE"},
		{3, "ALIGN_TARGET_MODE_USER_SPECIFIED_FRAME_OF_REFERENCE"},
	};
	static const hkInternalClassEnum hkbAlignBoneModifierEnums[] = {
		{"AlignModeABAM", hkbAlignBoneModifierAlignModeABAMEnumItems, 2, HK_NULL, 0 },
		{"AlignTargetMode", hkbAlignBoneModifierAlignTargetModeEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkbAlignBoneModifierAlignModeABAMEnum = reinterpret_cast<const hkClassEnum*>(&hkbAlignBoneModifierEnums[0]);
	const hkClassEnum* hkbAlignBoneModifierAlignTargetModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbAlignBoneModifierEnums[1]);
	static hkInternalClassMember hkbAlignBoneModifierClass_Members[] =
	{
		{ "alignMode", HK_NULL, hkbAlignBoneModifierAlignModeABAMEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "alignTargetMode", HK_NULL, hkbAlignBoneModifierAlignTargetModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "alignSingleAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "alignAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "alignTargetAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "frameOfReference", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "duration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "alignModeIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "alignTargetModeIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeSinceLastModify", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbAlignBoneModifierClass(
		"hkbAlignBoneModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbAlignBoneModifierEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbAlignBoneModifierClass_Members),
		HK_COUNT_OF(hkbAlignBoneModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbBalanceControllerModifierClass_Members[] =
	{
		{ "proportionalGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "derivativeGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "integralGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "checkBalanceModifier", &hkbCheckBalanceModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "boneForces", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL },
		{ "timestep", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "previousErrorMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "accumulatedErrorMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbBalanceControllerModifierClass(
		"hkbBalanceControllerModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbBalanceControllerModifierClass_Members),
		HK_COUNT_OF(hkbBalanceControllerModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbBalanceModifier_StepInfoClass_Members[] =
	{
		{ "boneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fractionOfSolution", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkbBalanceModifierStepInfoClass(
		"hkbBalanceModifierStepInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbBalanceModifier_StepInfoClass_Members),
		HK_COUNT_OF(hkbBalanceModifier_StepInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbBalanceModifierClass_Members[] =
	{
		{ "giveUp", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "comDistThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "passThrough", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollLeftFootBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollRightFootBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "balanceOnAnklesFraction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "upAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeInTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "comBiasX", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stepInfo", &hkbBalanceModifierStepInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "timeLapsed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbBalanceModifierClass(
		"hkbBalanceModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbBalanceModifierClass_Members),
		HK_COUNT_OF(hkbBalanceModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbCheckBalanceModifierClass_Members[] =
	{
		{ "ragdollLeftFootBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollRightFootBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "balanceOnAnklesFraction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "eventToSendWhenOffBalance", &hkbEventClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "offBalanceEventThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "worldUpAxisIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "comBiasX", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extractRagdollPose", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "mass", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "comWS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "desiredComWS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "offBalanceDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "errorMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbCheckBalanceModifierClass(
		"hkbCheckBalanceModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbCheckBalanceModifierClass_Members),
		HK_COUNT_OF(hkbCheckBalanceModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbCatchFallModifierFadeStateEnumItems[] =
	{
		{0, "FADE_IN"},
		{1, "FADE_OUT"},
	};
	static const hkInternalClassEnum hkbCatchFallModifierEnums[] = {
		{"FadeState", hkbCatchFallModifierFadeStateEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbCatchFallModifierFadeStateEnum = reinterpret_cast<const hkClassEnum*>(&hkbCatchFallModifierEnums[0]);
	static hkInternalClassMember hkbCatchFallModifier_HandClass_Members[] =
	{
		{ "handIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "animShoulderIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollShoulderIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollAnkleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbCatchFallModifierHand_DefaultStruct
		{
			int s_defaultOffsets[4];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt16 m_handIndex;
			hkInt16 m_animShoulderIndex;
			hkInt16 m_ragdollShoulderIndex;
			hkInt16 m_ragdollAnkleIndex;
		};
		const hkbCatchFallModifierHand_DefaultStruct hkbCatchFallModifierHand_Default =
		{
			{HK_OFFSET_OF(hkbCatchFallModifierHand_DefaultStruct,m_handIndex),HK_OFFSET_OF(hkbCatchFallModifierHand_DefaultStruct,m_animShoulderIndex),HK_OFFSET_OF(hkbCatchFallModifierHand_DefaultStruct,m_ragdollShoulderIndex),HK_OFFSET_OF(hkbCatchFallModifierHand_DefaultStruct,m_ragdollAnkleIndex)},
			-1,-1,-1,-1
		};
	}
	hkClass hkbCatchFallModifierHandClass(
		"hkbCatchFallModifierHand",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbCatchFallModifier_HandClass_Members),
		HK_COUNT_OF(hkbCatchFallModifier_HandClass_Members),
		&hkbCatchFallModifierHand_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbCatchFallModifierClass_Members[] =
	{
		{ "directionOfFallForwardLS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "directionOfFallRightLS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "directionOfFallUpLS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "spineIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT16, 0, 0, 0, HK_NULL },
		{ "leftHand", &hkbCatchFallModifierHandClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rightHand", &hkbCatchFallModifierHandClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "spreadHandsMultiplier", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "radarRange", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "previousTargetBlendWeight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "handsBendDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxReachDistanceForward", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxReachDistanceBackward", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeInReachGainSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeOutReachGainSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeOutDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeInTwistSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeOutTwistSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "catchFallDoneEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "raycastLayer", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "velocityRagdollBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "directionOfFallRagdollBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "orientHands", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeState", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "catchFallPosInBS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "currentReachGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "timeSinceLastModify", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentTwistGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentTwistDirection", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "catchFallPosIsValid", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "catchFallBegin", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "catchFallEnd", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbCatchFallModifier_DefaultStruct
		{
			int s_defaultOffsets[31];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_directionOfFallForwardLS;
			_hkVector4 m_directionOfFallRightLS;
			_hkVector4 m_directionOfFallUpLS;
			hkReal m_spreadHandsMultiplier;
			hkReal m_radarRange;
			hkReal m_previousTargetBlendWeight;
			hkReal m_handsBendDistance;
			hkReal m_maxReachDistanceForward;
			hkReal m_maxReachDistanceBackward;
			hkReal m_fadeInReachGainSpeed;
			hkReal m_fadeOutReachGainSpeed;
			hkReal m_fadeInTwistSpeed;
			hkReal m_fadeOutTwistSpeed;
			hkInt32 m_catchFallDoneEventId;
			hkInt16 m_raycastLayer;
			hkInt16 m_velocityRagdollBoneIndex;
			hkInt16 m_directionOfFallRagdollBoneIndex;
			_hkBool m_orientHands;
		};
		const hkbCatchFallModifier_DefaultStruct hkbCatchFallModifier_Default =
		{
			{HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_directionOfFallForwardLS),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_directionOfFallRightLS),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_directionOfFallUpLS),-1,-1,-1,HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_spreadHandsMultiplier),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_radarRange),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_previousTargetBlendWeight),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_handsBendDistance),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_maxReachDistanceForward),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_maxReachDistanceBackward),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_fadeInReachGainSpeed),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_fadeOutReachGainSpeed),-1,HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_fadeInTwistSpeed),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_fadeOutTwistSpeed),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_catchFallDoneEventId),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_raycastLayer),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_velocityRagdollBoneIndex),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_directionOfFallRagdollBoneIndex),HK_OFFSET_OF(hkbCatchFallModifier_DefaultStruct,m_orientHands),-1,-1,-1,-1,-1,-1,-1,-1,-1},
		{0,1,0},	{0,0,1},	{1,0,0},1,1.0f,0.75f,0.1f,0.6f,0.8f,1.0f,1.0f,1.0f,1.0f,-1,-1,-1,-1,true
		};
	}
	hkClass hkbCatchFallModifierClass(
		"hkbCatchFallModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbCatchFallModifierEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbCatchFallModifierClass_Members),
		HK_COUNT_OF(hkbCatchFallModifierClass_Members),
		&hkbCatchFallModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbCheckRagdollSpeedModifierClass_Members[] =
	{
		{ "minSpeedThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxSpeedThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "eventToSend", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbCheckRagdollSpeedModifier_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_eventToSend;
		};
		const hkbCheckRagdollSpeedModifier_DefaultStruct hkbCheckRagdollSpeedModifier_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkbCheckRagdollSpeedModifier_DefaultStruct,m_eventToSend)},
			-1
		};
	}
	hkClass hkbCheckRagdollSpeedModifierClass(
		"hkbCheckRagdollSpeedModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbCheckRagdollSpeedModifierClass_Members),
		HK_COUNT_OF(hkbCheckRagdollSpeedModifierClass_Members),
		&hkbCheckRagdollSpeedModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbComputeWorldFromModelModifierClass_Members[] =
	{
		{ "poseMatchingRootBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "poseMatchingOtherBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "poseMatchingAnotherBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbComputeWorldFromModelModifier_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt16 m_poseMatchingRootBoneIndex;
			hkInt16 m_poseMatchingOtherBoneIndex;
			hkInt16 m_poseMatchingAnotherBoneIndex;
		};
		const hkbComputeWorldFromModelModifier_DefaultStruct hkbComputeWorldFromModelModifier_Default =
		{
			{HK_OFFSET_OF(hkbComputeWorldFromModelModifier_DefaultStruct,m_poseMatchingRootBoneIndex),HK_OFFSET_OF(hkbComputeWorldFromModelModifier_DefaultStruct,m_poseMatchingOtherBoneIndex),HK_OFFSET_OF(hkbComputeWorldFromModelModifier_DefaultStruct,m_poseMatchingAnotherBoneIndex)},
			-1,-1,-1
		};
	}
	hkClass hkbComputeWorldFromModelModifierClass(
		"hkbComputeWorldFromModelModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbComputeWorldFromModelModifierClass_Members),
		HK_COUNT_OF(hkbComputeWorldFromModelModifierClass_Members),
		&hkbComputeWorldFromModelModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbDetectCloseToGroundModifierClass_Members[] =
	{
		{ "closeToGroundEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "closeToGroundHeight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "raycastDistanceDown", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionFilterInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "raycastInterface", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isCloseToGround", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbDetectCloseToGroundModifier_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_closeToGroundEventId;
			hkReal m_closeToGroundHeight;
			hkReal m_raycastDistanceDown;
			hkUint32 m_collisionFilterInfo;
			hkInt16 m_boneIndex;
		};
		const hkbDetectCloseToGroundModifier_DefaultStruct hkbDetectCloseToGroundModifier_Default =
		{
			{HK_OFFSET_OF(hkbDetectCloseToGroundModifier_DefaultStruct,m_closeToGroundEventId),HK_OFFSET_OF(hkbDetectCloseToGroundModifier_DefaultStruct,m_closeToGroundHeight),HK_OFFSET_OF(hkbDetectCloseToGroundModifier_DefaultStruct,m_raycastDistanceDown),HK_OFFSET_OF(hkbDetectCloseToGroundModifier_DefaultStruct,m_collisionFilterInfo),HK_OFFSET_OF(hkbDetectCloseToGroundModifier_DefaultStruct,m_boneIndex),-1,-1},
			-1,0.5,0.8f,3,-1
		};
	}
	hkClass hkbDetectCloseToGroundModifierClass(
		"hkbDetectCloseToGroundModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbDetectCloseToGroundModifierClass_Members),
		HK_COUNT_OF(hkbDetectCloseToGroundModifierClass_Members),
		&hkbDetectCloseToGroundModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbGravityModifierClass_Members[] =
	{
		{ "initialVelocityInMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "gravityConstant", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentVelocityInMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "secondsElapsed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbGravityModifier_DefaultStruct
		{
			int s_defaultOffsets[4];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_initialVelocityInMS;
			hkReal m_gravityConstant;
		};
		const hkbGravityModifier_DefaultStruct hkbGravityModifier_Default =
		{
			{HK_OFFSET_OF(hkbGravityModifier_DefaultStruct,m_initialVelocityInMS),HK_OFFSET_OF(hkbGravityModifier_DefaultStruct,m_gravityConstant),-1,-1},
			{0,0,0},9.8f
		};
	}
	hkClass hkbGravityModifierClass(
		"hkbGravityModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbGravityModifierClass_Members),
		HK_COUNT_OF(hkbGravityModifierClass_Members),
		&hkbGravityModifier_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbReachModifierReachModeEnumItems[] =
	{
		{0, "REACH_MODE_TERRAIN"},
		{1, "REACH_MODE_WORLD_POSITION"},
		{2, "REACH_MODE_MODEL_POSITION"},
		{3, "REACH_MODE_BONE_POSITION"},
	};
	static const hkInternalClassEnum hkbReachModifierEnums[] = {
		{"ReachMode", hkbReachModifierReachModeEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkbReachModifierReachModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbReachModifierEnums[0]);
	static hkInternalClassMember hkbReachModifier_HandClass_Members[] =
	{
		{ "targetOrSensingPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetBackHandNormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensingRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "handIkTrackIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbReachModifierHand_DefaultStruct
		{
			int s_defaultOffsets[5];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_targetOrSensingPosition;
			_hkVector4 m_targetBackHandNormal;
			hkReal m_sensingRadius;
			hkInt16 m_boneIndex;
			hkInt16 m_handIkTrackIndex;
		};
		const hkbReachModifierHand_DefaultStruct hkbReachModifierHand_Default =
		{
			{HK_OFFSET_OF(hkbReachModifierHand_DefaultStruct,m_targetOrSensingPosition),HK_OFFSET_OF(hkbReachModifierHand_DefaultStruct,m_targetBackHandNormal),HK_OFFSET_OF(hkbReachModifierHand_DefaultStruct,m_sensingRadius),HK_OFFSET_OF(hkbReachModifierHand_DefaultStruct,m_boneIndex),HK_OFFSET_OF(hkbReachModifierHand_DefaultStruct,m_handIkTrackIndex)},
		{0,0,0},	{0,0,0},0.4f,-1,-1
		};
	}
	hkClass hkbReachModifierHandClass(
		"hkbReachModifierHand",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbReachModifier_HandClass_Members),
		HK_COUNT_OF(hkbReachModifier_HandClass_Members),
		&hkbReachModifierHand_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbReachModifierClass_Members[] =
	{
		{ "hands", &hkbReachModifierHandClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "newTargetGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "noTargetGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeOutDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "raycastLayer", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensingPropertyKey", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "reachMode", HK_NULL, hkbReachModifierReachModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "ignoreMySystemGroup", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extrapolate", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "raycastInterface", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "internalHandData", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeLapse", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbReachModifier_DefaultStruct
		{
			int s_defaultOffsets[13];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_newTargetGain;
			hkReal m_noTargetGain;
			hkReal m_targetGain;
			hkInt32 m_raycastLayer;
			_hkBool m_ignoreMySystemGroup;
			hkReal m_extrapolate;
		};
		const hkbReachModifier_DefaultStruct hkbReachModifier_Default =
		{
			{-1,HK_OFFSET_OF(hkbReachModifier_DefaultStruct,m_newTargetGain),HK_OFFSET_OF(hkbReachModifier_DefaultStruct,m_noTargetGain),HK_OFFSET_OF(hkbReachModifier_DefaultStruct,m_targetGain),-1,HK_OFFSET_OF(hkbReachModifier_DefaultStruct,m_raycastLayer),-1,-1,HK_OFFSET_OF(hkbReachModifier_DefaultStruct,m_ignoreMySystemGroup),HK_OFFSET_OF(hkbReachModifier_DefaultStruct,m_extrapolate),-1,-1,-1},
			0.085f,0.19f,0.3f,-1,true,1.0f
		};
	}
	hkClass hkbReachModifierClass(
		"hkbReachModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbReachModifierEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbReachModifierClass_Members),
		HK_COUNT_OF(hkbReachModifierClass_Members),
		&hkbReachModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbStateDependentModifierClass_Members[] =
	{
		{ "applyModifierDuringTransition", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stateIds", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "modifier", &hkbModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "isActive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stateMachine", &hkbStateMachineClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkbStateDependentModifierClass(
		"hkbStateDependentModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbStateDependentModifierClass_Members),
		HK_COUNT_OF(hkbStateDependentModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbConstrainRigidBodyModifierPivotPlacementEnumItems[] =
	{
		{0, "PIVOT_MIDWAY_BETWEEN_CENTROIDS"},
		{1, "PIVOT_AT_TARGET_CONTACT_POINT"},
		{2, "PIVOT_MIDWAY_BETWEEN_TARGET_SHAPE_CENTROID_AND_BODY_TO_CONSTRAIN_CENTROID"},
	};
	static const hkInternalClassEnumItem hkbConstrainRigidBodyModifierBoneToConstrainPlacementEnumItems[] =
	{
		{0, "BTCP_AT_CURRENT_POSITION"},
		{1, "BTCP_ALIGN_COM_AND_PIVOT"},
	};
	static const hkInternalClassEnum hkbConstrainRigidBodyModifierEnums[] = {
		{"PivotPlacement", hkbConstrainRigidBodyModifierPivotPlacementEnumItems, 3, HK_NULL, 0 },
		{"BoneToConstrainPlacement", hkbConstrainRigidBodyModifierBoneToConstrainPlacementEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbConstrainRigidBodyModifierPivotPlacementEnum = reinterpret_cast<const hkClassEnum*>(&hkbConstrainRigidBodyModifierEnums[0]);
	const hkClassEnum* hkbConstrainRigidBodyModifierBoneToConstrainPlacementEnum = reinterpret_cast<const hkClassEnum*>(&hkbConstrainRigidBodyModifierEnums[1]);
	static hkInternalClassMember hkbConstrainRigidBodyModifierClass_Members[] =
	{
		{ "breakThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "target", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollBoneToConstrain", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "breakable", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pivotPlacement", HK_NULL, hkbConstrainRigidBodyModifierPivotPlacementEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "boneToConstrainPlacement", HK_NULL, hkbConstrainRigidBodyModifierBoneToConstrainPlacementEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "constraintType", HK_NULL, hkpConstraintDataConstraintTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "clearTargetData", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isConstraintHinge", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "constraint", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "behaviorTarget", &hkbTargetClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbConstrainRigidBodyModifier_DefaultStruct
		{
			int s_defaultOffsets[11];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_breakThreshold;
			hkInt16 m_ragdollBoneToConstrain;
			_hkBool m_isConstraintHinge;
		};
		const hkbConstrainRigidBodyModifier_DefaultStruct hkbConstrainRigidBodyModifier_Default =
		{
			{HK_OFFSET_OF(hkbConstrainRigidBodyModifier_DefaultStruct,m_breakThreshold),-1,HK_OFFSET_OF(hkbConstrainRigidBodyModifier_DefaultStruct,m_ragdollBoneToConstrain),-1,-1,-1,-1,-1,HK_OFFSET_OF(hkbConstrainRigidBodyModifier_DefaultStruct,m_isConstraintHinge),-1,-1},
			1.0f,-1,true
		};
	}
	hkClass hkbConstrainRigidBodyModifierClass(
		"hkbConstrainRigidBodyModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbConstrainRigidBodyModifierEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbConstrainRigidBodyModifierClass_Members),
		HK_COUNT_OF(hkbConstrainRigidBodyModifierClass_Members),
		&hkbConstrainRigidBodyModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbControlledReachModifierClass_Members[] =
	{
		{ "target", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "fadeInStart", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeInEnd", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeOutStart", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeOutEnd", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeOutDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensorAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "handIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "isHandEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL }
	};
	hkClass hkbControlledReachModifierClass(
		"hkbControlledReachModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbControlledReachModifierClass_Members),
		HK_COUNT_OF(hkbControlledReachModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbFaceTargetModifierClass_Members[] =
	{
		{ "offsetAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "target", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "onlyOnce", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "behaviorTarget", &hkbTargetClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "done", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbFaceTargetModifierClass(
		"hkbFaceTargetModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbFaceTargetModifierClass_Members),
		HK_COUNT_OF(hkbFaceTargetModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbMoveBoneTowardTargetModifierTargetModeMBTTEnumItems[] =
	{
		{0, "TARGET_POSITION"},
		{1, "TARGET_COM"},
		{2, "TARGET_CONTACT_POINT"},
		{3, "TARGET_SHAPE_CENTROID"},
	};
	static const hkInternalClassEnumItem hkbMoveBoneTowardTargetModifierAlignModeBitsEnumItems[] =
	{
		{1/*0x1*/, "ALIGN_AXES"},
		{2/*0x2*/, "ALIGN_BONE_AXIS_WITH_CONTACT_NORMAL"},
		{4/*0x4*/, "ALIGN_WITH_CHARACTER_FORWARD"},
	};
	static const hkInternalClassEnum hkbMoveBoneTowardTargetModifierEnums[] = {
		{"TargetModeMBTT", hkbMoveBoneTowardTargetModifierTargetModeMBTTEnumItems, 4, HK_NULL, 0 },
		{"AlignModeBits", hkbMoveBoneTowardTargetModifierAlignModeBitsEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkbMoveBoneTowardTargetModifierTargetModeMBTTEnum = reinterpret_cast<const hkClassEnum*>(&hkbMoveBoneTowardTargetModifierEnums[0]);
	const hkClassEnum* hkbMoveBoneTowardTargetModifierAlignModeBitsEnum = reinterpret_cast<const hkClassEnum*>(&hkbMoveBoneTowardTargetModifierEnums[1]);
	static hkInternalClassMember hkbMoveBoneTowardTargetModifierClass_Members[] =
	{
		{ "offsetInBoneSpace", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "alignAxisBS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetAlignAxisTS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "alignWithCharacterForwardBS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentBonePositionOut", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentBoneRotationOut", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "childGenerator", &hkbGeneratorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "duration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "eventToSendWhenTargetReached", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "target", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "animationBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetMode", HK_NULL, hkbMoveBoneTowardTargetModifierTargetModeMBTTEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "alignMode", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useVelocityPrediction", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "clearTargetData", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "affectOrientation", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentBoneIsValidOut", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentBoneIsValidIn", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetInternal", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "targetPointTS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeSinceLastModify", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "behaviorTarget", &hkbTargetClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "lastAnimBonePositionMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "finalAnimBonePositionMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initialAnimBonePositionMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "finalAnimBoneOrientationMS", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "animationFromRagdoll", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "totalMotion", HK_NULL, HK_NULL, hkClassMember::TYPE_QSTRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "accumulatedMotion", HK_NULL, HK_NULL, hkClassMember::TYPE_QSTRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useAnimationData", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbMoveBoneTowardTargetModifier_DefaultStruct
		{
			int s_defaultOffsets[32];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_offsetInBoneSpace;
			_hkVector4 m_alignAxisBS;
			_hkVector4 m_targetAlignAxisTS;
			_hkVector4 m_alignWithCharacterForwardBS;
			_hkVector4 m_currentBonePositionOut;
			_hkQuaternion m_currentBoneRotationOut;
			hkReal m_duration;
			hkInt32 m_eventToSendWhenTargetReached;
			hkInt16 m_ragdollBoneIndex;
			hkInt16 m_animationBoneIndex;
			_hkBool m_affectOrientation;
		};
		const hkbMoveBoneTowardTargetModifier_DefaultStruct hkbMoveBoneTowardTargetModifier_Default =
		{
			{HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_offsetInBoneSpace),HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_alignAxisBS),HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_targetAlignAxisTS),HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_alignWithCharacterForwardBS),HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_currentBonePositionOut),HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_currentBoneRotationOut),-1,HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_duration),HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_eventToSendWhenTargetReached),-1,HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_ragdollBoneIndex),HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_animationBoneIndex),-1,-1,-1,-1,HK_OFFSET_OF(hkbMoveBoneTowardTargetModifier_DefaultStruct,m_affectOrientation),-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
		{0,1,0},	{0,1,0},	{0,1,0},	{0,1,0},	{0,0,0},	{0,0,0,1},1,-1,-1,-1,true
		};
	}
	hkClass hkbMoveBoneTowardTargetModifierClass(
		"hkbMoveBoneTowardTargetModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbMoveBoneTowardTargetModifierEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkbMoveBoneTowardTargetModifierClass_Members),
		HK_COUNT_OF(hkbMoveBoneTowardTargetModifierClass_Members),
		&hkbMoveBoneTowardTargetModifier_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbReachTowardTargetModifierFadeStateEnumItems[] =
	{
		{0, "FADE_IN"},
		{1, "FADE_OUT"},
	};
	static const hkInternalClassEnum hkbReachTowardTargetModifierEnums[] = {
		{"FadeState", hkbReachTowardTargetModifierFadeStateEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbReachTowardTargetModifierFadeStateEnum = reinterpret_cast<const hkClassEnum*>(&hkbReachTowardTargetModifierEnums[0]);
	static hkInternalClassMember hkbReachTowardTargetModifier_HandClass_Members[] =
	{
		{ "handIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "shoulderIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbReachTowardTargetModifierHand_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt16 m_handIndex;
			hkInt16 m_shoulderIndex;
			_hkBool m_isEnabled;
		};
		const hkbReachTowardTargetModifierHand_DefaultStruct hkbReachTowardTargetModifierHand_Default =
		{
			{HK_OFFSET_OF(hkbReachTowardTargetModifierHand_DefaultStruct,m_handIndex),HK_OFFSET_OF(hkbReachTowardTargetModifierHand_DefaultStruct,m_shoulderIndex),HK_OFFSET_OF(hkbReachTowardTargetModifierHand_DefaultStruct,m_isEnabled)},
			-1,-1,true
		};
	}
	hkClass hkbReachTowardTargetModifierHandClass(
		"hkbReachTowardTargetModifierHand",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbReachTowardTargetModifier_HandClass_Members),
		HK_COUNT_OF(hkbReachTowardTargetModifier_HandClass_Members),
		&hkbReachTowardTargetModifierHand_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbReachTowardTargetModifierClass_Members[] =
	{
		{ "leftHand", &hkbReachTowardTargetModifierHandClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rightHand", &hkbReachTowardTargetModifierHandClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "distanceBetweenHands", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "reachDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeInGainSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeOutGainSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fadeOutDuration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetChangeSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "target", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "holdTarget", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "reachPastTarget", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "giveUpIfNoTarget", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "clearTargetData", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "behaviorTarget", &hkbTargetClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "targetRB", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "prevTargetInMS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "currentGain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "timeSinceLastModify", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "fadeState", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "haveGivenUp", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isTherePrevTarget", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbReachTowardTargetModifier_DefaultStruct
		{
			int s_defaultOffsets[21];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_distanceBetweenHands;
			hkReal m_reachDistance;
			hkReal m_fadeInGainSpeed;
			hkReal m_fadeOutGainSpeed;
			hkReal m_targetChangeSpeed;
			_hkBool m_reachPastTarget;
		};
		const hkbReachTowardTargetModifier_DefaultStruct hkbReachTowardTargetModifier_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkbReachTowardTargetModifier_DefaultStruct,m_distanceBetweenHands),HK_OFFSET_OF(hkbReachTowardTargetModifier_DefaultStruct,m_reachDistance),HK_OFFSET_OF(hkbReachTowardTargetModifier_DefaultStruct,m_fadeInGainSpeed),HK_OFFSET_OF(hkbReachTowardTargetModifier_DefaultStruct,m_fadeOutGainSpeed),-1,HK_OFFSET_OF(hkbReachTowardTargetModifier_DefaultStruct,m_targetChangeSpeed),-1,-1,HK_OFFSET_OF(hkbReachTowardTargetModifier_DefaultStruct,m_reachPastTarget),-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
			0.3f,0.5f,1.0f,1.0f,1.0f,true
		};
	}
	hkClass hkbReachTowardTargetModifierClass(
		"hkbReachTowardTargetModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbReachTowardTargetModifierEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkbReachTowardTargetModifierClass_Members),
		HK_COUNT_OF(hkbReachTowardTargetModifierClass_Members),
		&hkbReachTowardTargetModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbTargetClass_Members[] =
	{
		{ "target", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "contactPointTS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "contactNormalTS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "shapeCentroidTS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "distance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "targetPriority", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkbTargetClass(
		"hkbTarget",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbTargetClass_Members),
		HK_COUNT_OF(hkbTargetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkbTargetRigidBodyModifierEventModeTRBAMEnumItems[] =
	{
		{0, "EVENT_MODE_DO_NOT_SEND"},
		{1, "EVENT_MODE_SEND_ONCE"},
		{2, "EVENT_MODE_RESEND"},
	};
	static const hkInternalClassEnumItem hkbTargetRigidBodyModifierTargetModeEnumItems[] =
	{
		{0, "TARGET_MODE_CONE_AROUND_CHARACTER_FORWARD"},
		{1, "TARGET_MODE_CONE_AROUND_LOCAL_AXIS"},
		{2, "TARGET_MODE_RAYCAST_ALONG_CHARACTER_FORWARD"},
		{3, "TARGET_MODE_RAYCAST_ALONG_LOCAL_AXIS"},
		{4, "TARGET_MODE_ANY_DIRECTION"},
	};
	static const hkInternalClassEnumItem hkbTargetRigidBodyModifierComputeTargetAngleModeEnumItems[] =
	{
		{0, "COMPUTE_ANGLE_USING_TARGET_COM"},
		{1, "COMPUTE_ANGLE_USING_TARGET_CONTACT_POINT"},
	};
	static const hkInternalClassEnumItem hkbTargetRigidBodyModifierComputeTargetDistanceModeEnumItems[] =
	{
		{0, "COMPUTE_DISTANCE_USING_TARGET_POSITION"},
		{1, "COMPUTE_DISTANCE_USING_TARGET_CONTACT_POINT"},
	};
	static const hkInternalClassEnum hkbTargetRigidBodyModifierEnums[] = {
		{"EventModeTRBAM", hkbTargetRigidBodyModifierEventModeTRBAMEnumItems, 3, HK_NULL, 0 },
		{"TargetMode", hkbTargetRigidBodyModifierTargetModeEnumItems, 5, HK_NULL, 0 },
		{"ComputeTargetAngleMode", hkbTargetRigidBodyModifierComputeTargetAngleModeEnumItems, 2, HK_NULL, 0 },
		{"ComputeTargetDistanceMode", hkbTargetRigidBodyModifierComputeTargetDistanceModeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkbTargetRigidBodyModifierEventModeTRBAMEnum = reinterpret_cast<const hkClassEnum*>(&hkbTargetRigidBodyModifierEnums[0]);
	const hkClassEnum* hkbTargetRigidBodyModifierTargetModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbTargetRigidBodyModifierEnums[1]);
	const hkClassEnum* hkbTargetRigidBodyModifierComputeTargetAngleModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbTargetRigidBodyModifierEnums[2]);
	const hkClassEnum* hkbTargetRigidBodyModifierComputeTargetDistanceModeEnum = reinterpret_cast<const hkClassEnum*>(&hkbTargetRigidBodyModifierEnums[3]);
	static hkInternalClassMember hkbTargetRigidBodyModifierClass_Members[] =
	{
		{ "targetMode", HK_NULL, hkbTargetRigidBodyModifierTargetModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "sensingLayer", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetOnlyOnce", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ignoreMySystemGroup", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxTargetDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxTargetHeightAboveSensor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "closeToTargetDistanceThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetAngleMode", HK_NULL, hkbTargetRigidBodyModifierComputeTargetAngleModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "targetDistanceMode", HK_NULL, hkbTargetRigidBodyModifierComputeTargetDistanceModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "maxAngleToTarget", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensorRagdollBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensorAnimationBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "closeToTargetRagdollBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "closeToTargetAnimationBoneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensorOffsetInBoneSpace", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "closeToTargetOffsetInBoneSpace", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensorDirectionBS", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "eventMode", HK_NULL, hkbTargetRigidBodyModifierEventModeTRBAMEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "sensingPropertyKey", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensorInWS", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "eventToSend", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "eventToSendToTarget", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "closeToTargetEventId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "target", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetIn", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useVelocityPrediction", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetOnlySpheres", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetPriority", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "behaviorTarget", &hkbTargetClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "timeSinceLastModify", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "eventHasBeenSent", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "closeToTargetEventHasBeenSent", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isActive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isTargetMemoryAllocatedInternally", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbTargetRigidBodyModifier_DefaultStruct
		{
			int s_defaultOffsets[34];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_sensingLayer;
			_hkBool m_targetOnlyOnce;
			_hkBool m_ignoreMySystemGroup;
			hkReal m_maxTargetDistance;
			hkReal m_maxTargetHeightAboveSensor;
			hkReal m_closeToTargetDistanceThreshold;
			hkReal m_maxAngleToTarget;
			hkInt16 m_sensorRagdollBoneIndex;
			hkInt16 m_sensorAnimationBoneIndex;
			hkInt16 m_closeToTargetRagdollBoneIndex;
			hkInt16 m_closeToTargetAnimationBoneIndex;
			_hkVector4 m_sensorOffsetInBoneSpace;
			_hkVector4 m_closeToTargetOffsetInBoneSpace;
			_hkVector4 m_sensorDirectionBS;
			hkInt32 m_eventToSend;
			hkInt32 m_eventToSendToTarget;
			hkInt32 m_closeToTargetEventId;
		};
		const hkbTargetRigidBodyModifier_DefaultStruct hkbTargetRigidBodyModifier_Default =
		{
			{-1,HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_sensingLayer),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_targetOnlyOnce),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_ignoreMySystemGroup),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_maxTargetDistance),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_maxTargetHeightAboveSensor),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_closeToTargetDistanceThreshold),-1,-1,HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_maxAngleToTarget),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_sensorRagdollBoneIndex),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_sensorAnimationBoneIndex),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_closeToTargetRagdollBoneIndex),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_closeToTargetAnimationBoneIndex),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_sensorOffsetInBoneSpace),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_closeToTargetOffsetInBoneSpace),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_sensorDirectionBS),-1,-1,-1,HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_eventToSend),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_eventToSendToTarget),HK_OFFSET_OF(hkbTargetRigidBodyModifier_DefaultStruct,m_closeToTargetEventId),-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	-1,true,true,1.0f,2.0f,0.1f,1.0f,-1,-1,-1,-1,	{0,1,0},	{0,1,0},	{0,1,0},-1,-1,-1
		};
	}
	hkClass hkbTargetRigidBodyModifierClass(
		"hkbTargetRigidBodyModifier",
		&hkbModifierClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkbTargetRigidBodyModifierEnums),
		4,
		reinterpret_cast<const hkClassMember*>(hkbTargetRigidBodyModifierClass_Members),
		HK_COUNT_OF(hkbTargetRigidBodyModifierClass_Members),
		&hkbTargetRigidBodyModifier_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbClimbMountingPredicateClass_Members[] =
	{
		{ "maxTargetDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sensingForLeftHand", &hkbTargetRigidBodyModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "sensingForRightHand", &hkbTargetRigidBodyModifierClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "targetForLeftHand", &hkbTargetClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "targetForRightHand", &hkbTargetClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkbClimbMountingPredicate_DefaultStruct
		{
			int s_defaultOffsets[5];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_maxTargetDistance;
		};
		const hkbClimbMountingPredicate_DefaultStruct hkbClimbMountingPredicate_Default =
		{
			{HK_OFFSET_OF(hkbClimbMountingPredicate_DefaultStruct,m_maxTargetDistance),-1,-1,-1,-1},
			1.0f
		};
	}
	hkClass hkbClimbMountingPredicateClass(
		"hkbClimbMountingPredicate",
		&hkbPredicateClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbClimbMountingPredicateClass_Members),
		HK_COUNT_OF(hkbClimbMountingPredicateClass_Members),
		&hkbClimbMountingPredicate_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkbHoldFromBlendingTransitionEffectClass_Members[] =
	{
		{ "heldFromPose", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "heldFromPoseSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "heldWorldFromModel", HK_NULL, HK_NULL, hkClassMember::TYPE_QSTRANSFORM, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "heldFromSkeleton", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "isFromGeneratorActive", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkbHoldFromBlendingTransitionEffect_DefaultStruct
		{
			int s_defaultOffsets[5];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_isFromGeneratorActive;
		};
		const hkbHoldFromBlendingTransitionEffect_DefaultStruct hkbHoldFromBlendingTransitionEffect_Default =
		{
			{-1,-1,-1,-1,HK_OFFSET_OF(hkbHoldFromBlendingTransitionEffect_DefaultStruct,m_isFromGeneratorActive)},
			true
		};
	}
	hkClass hkbHoldFromBlendingTransitionEffectClass(
		"hkbHoldFromBlendingTransitionEffect",
		&hkbBlendingTransitionEffectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkbHoldFromBlendingTransitionEffectClass_Members),
		HK_COUNT_OF(hkbHoldFromBlendingTransitionEffectClass_Members),
		&hkbHoldFromBlendingTransitionEffect_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkVariableTweakingHelper_BoolVariableInfoClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tweakOn", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkVariableTweakingHelperBoolVariableInfoClass(
		"hkVariableTweakingHelperBoolVariableInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkVariableTweakingHelper_BoolVariableInfoClass_Members),
		HK_COUNT_OF(hkVariableTweakingHelper_BoolVariableInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkVariableTweakingHelper_IntVariableInfoClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tweakOn", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkVariableTweakingHelperIntVariableInfoClass(
		"hkVariableTweakingHelperIntVariableInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkVariableTweakingHelper_IntVariableInfoClass_Members),
		HK_COUNT_OF(hkVariableTweakingHelper_IntVariableInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkVariableTweakingHelper_RealVariableInfoClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tweakOn", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkVariableTweakingHelperRealVariableInfoClass(
		"hkVariableTweakingHelperRealVariableInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkVariableTweakingHelper_RealVariableInfoClass_Members),
		HK_COUNT_OF(hkVariableTweakingHelper_RealVariableInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkVariableTweakingHelper_Vector4VariableInfoClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "x", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "y", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "z", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tweakOn", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkVariableTweakingHelperVector4VariableInfoClass(
		"hkVariableTweakingHelperVector4VariableInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkVariableTweakingHelper_Vector4VariableInfoClass_Members),
		HK_COUNT_OF(hkVariableTweakingHelper_Vector4VariableInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkVariableTweakingHelperClass_Members[] =
	{
		{ "boolVariableInfo", &hkVariableTweakingHelperBoolVariableInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "intVariableInfo", &hkVariableTweakingHelperIntVariableInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "realVariableInfo", &hkVariableTweakingHelperRealVariableInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "vector4VariableInfo", &hkVariableTweakingHelperVector4VariableInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "behavior", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "boolIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "intIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "realIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "vector4Indices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkVariableTweakingHelperClass(
		"hkVariableTweakingHelper",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkVariableTweakingHelperClass_Members),
		HK_COUNT_OF(hkVariableTweakingHelperClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hclActionClass(
		"hclAction",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclSimpleWindActionClass_Members[] =
	{
		{ "windDirection", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "windMinSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "windMaxSpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "windFrequency", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maximumDrag", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "airVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclSimpleWindActionClass(
		"hclSimpleWindAction",
		&hclActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclSimpleWindActionClass_Members),
		HK_COUNT_OF(hclSimpleWindActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclBufferDefinitionClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "subType", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numVertices", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numTriangles", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclBufferDefinitionClass(
		"hclBufferDefinition",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclBufferDefinitionClass_Members),
		HK_COUNT_OF(hclBufferDefinitionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclClothDataClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "simClothDatas", &hclSimClothDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "bufferDefinitions", &hclBufferDefinitionClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "transformSetDefinitions", &hclTransformSetDefinitionClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "operators", &hclOperatorClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "clothStateDatas", &hclClothStateClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "actions", &hclActionClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hclClothDataClass(
		"hclClothData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclClothDataClass_Members),
		HK_COUNT_OF(hclClothDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclCollidableClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transform", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "linearVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angularVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stepMovement", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "shape", &hclShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclCollidableClass(
		"hclCollidable",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclCollidableClass_Members),
		HK_COUNT_OF(hclCollidableClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclShapeClass_Members[] =
	{
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT32, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hclShapeClass(
		"hclShape",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclShapeClass_Members),
		HK_COUNT_OF(hclShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclCapsuleShapeClass_Members[] =
	{
		{ "start", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "end", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dir", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "capLenSqrd", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclCapsuleShapeClass(
		"hclCapsuleShape",
		&hclShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclCapsuleShapeClass_Members),
		HK_COUNT_OF(hclCapsuleShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclConvexHeightFieldShapeClass_Members[] =
	{
		{ "res", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "resIncBorder", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "floatCorrectionOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "heights", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "allocatedHeights", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "faces", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 6, 0, 0, HK_NULL },
		{ "localToMapTransform", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "localToMapScale", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclConvexHeightFieldShapeClass(
		"hclConvexHeightFieldShape",
		&hclShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclConvexHeightFieldShapeClass_Members),
		HK_COUNT_OF(hclConvexHeightFieldShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclCylinderShapeClass_Members[] =
	{
		{ "start", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "end", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dir", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "cylLenSqrd", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "radiusSqrd", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclCylinderShapeClass(
		"hclCylinderShape",
		&hclShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclCylinderShapeClass_Members),
		HK_COUNT_OF(hclCylinderShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclPlaneShapeClass_Members[] =
	{
		{ "planeEquation", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclPlaneShapeClass(
		"hclPlaneShape",
		&hclShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclPlaneShapeClass_Members),
		HK_COUNT_OF(hclPlaneShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclSphereShapeClass_Members[] =
	{
		{ "sphere", &hkSphereClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclSphereShapeClass(
		"hclSphereShape",
		&hclShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclSphereShapeClass_Members),
		HK_COUNT_OF(hclSphereShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hclConstraintSetMaxConstraintSetSizeEnumItems[] =
	{
		{128, "MAX_CONSTRAINT_SET_SIZE"},
	};
	static const hkInternalClassEnum hclConstraintSetEnums[] = {
		{"MaxConstraintSetSize", hclConstraintSetMaxConstraintSetSizeEnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hclConstraintSetMaxConstraintSetSizeEnum = reinterpret_cast<const hkClassEnum*>(&hclConstraintSetEnums[0]);
	static hkInternalClassMember hclConstraintSetClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hclConstraintSetClass(
		"hclConstraintSet",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hclConstraintSetEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hclConstraintSetClass_Members),
		HK_COUNT_OF(hclConstraintSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclBendLinkConstraintSet_LinkClass_Members[] =
	{
		{ "particleA", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particleB", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bendMinLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stretchMaxLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bendStiffness", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stretchStiffness", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclBendLinkConstraintSetLinkClass(
		"hclBendLinkConstraintSetLink",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclBendLinkConstraintSet_LinkClass_Members),
		HK_COUNT_OF(hclBendLinkConstraintSet_LinkClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclBendLinkConstraintSetClass_Members[] =
	{
		{ "links", &hclBendLinkConstraintSetLinkClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hclBendLinkConstraintSetClass(
		"hclBendLinkConstraintSet",
		&hclConstraintSetClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclBendLinkConstraintSetClass_Members),
		HK_COUNT_OF(hclBendLinkConstraintSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclBonePlanesConstraintSet_BonePlaneClass_Members[] =
	{
		{ "planeEquationBone", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transformIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stiffness", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclBonePlanesConstraintSetBonePlaneClass(
		"hclBonePlanesConstraintSetBonePlane",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclBonePlanesConstraintSet_BonePlaneClass_Members),
		HK_COUNT_OF(hclBonePlanesConstraintSet_BonePlaneClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclBonePlanesConstraintSetClass_Members[] =
	{
		{ "bonePlanes", &hclBonePlanesConstraintSetBonePlaneClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "transformSetIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclBonePlanesConstraintSetClass(
		"hclBonePlanesConstraintSet",
		&hclConstraintSetClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclBonePlanesConstraintSetClass_Members),
		HK_COUNT_OF(hclBonePlanesConstraintSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclHingeConstraintSet_HingeClass_Members[] =
	{
		{ "particleA", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particleB", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particle1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particle2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hingeStiffness", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "relaxFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sinHalfAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclHingeConstraintSetHingeClass(
		"hclHingeConstraintSetHinge",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclHingeConstraintSet_HingeClass_Members),
		HK_COUNT_OF(hclHingeConstraintSet_HingeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclHingeConstraintSetClass_Members[] =
	{
		{ "hinges", &hclHingeConstraintSetHingeClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hclHingeConstraintSetClass(
		"hclHingeConstraintSet",
		&hclConstraintSetClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclHingeConstraintSetClass_Members),
		HK_COUNT_OF(hclHingeConstraintSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hclLocalRangeConstraintSetForceUpgrade6EnumItems[] =
	{
		{0, "HCL_FORCE_UPGRADE6"},
	};
	static const hkInternalClassEnum hclLocalRangeConstraintSetEnums[] = {
		{"ForceUpgrade6", hclLocalRangeConstraintSetForceUpgrade6EnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hclLocalRangeConstraintSetForceUpgrade6Enum = reinterpret_cast<const hkClassEnum*>(&hclLocalRangeConstraintSetEnums[0]);
	static hkInternalClassMember hclLocalRangeConstraintSet_LocalConstraintClass_Members[] =
	{
		{ "particleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "referenceVertex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maximumDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxNormalDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minNormalDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclLocalRangeConstraintSetLocalConstraintClass(
		"hclLocalRangeConstraintSetLocalConstraint",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclLocalRangeConstraintSet_LocalConstraintClass_Members),
		HK_COUNT_OF(hclLocalRangeConstraintSet_LocalConstraintClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclLocalRangeConstraintSetClass_Members[] =
	{
		{ "localConstraints", &hclLocalRangeConstraintSetLocalConstraintClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "referenceMeshBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclLocalRangeConstraintSetClass(
		"hclLocalRangeConstraintSet",
		&hclConstraintSetClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hclLocalRangeConstraintSetEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hclLocalRangeConstraintSetClass_Members),
		HK_COUNT_OF(hclLocalRangeConstraintSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclStandardLinkConstraintSet_LinkClass_Members[] =
	{
		{ "particleA", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particleB", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "restLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stiffness", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclStandardLinkConstraintSetLinkClass(
		"hclStandardLinkConstraintSetLink",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclStandardLinkConstraintSet_LinkClass_Members),
		HK_COUNT_OF(hclStandardLinkConstraintSet_LinkClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclStandardLinkConstraintSetClass_Members[] =
	{
		{ "links", &hclStandardLinkConstraintSetLinkClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hclStandardLinkConstraintSetClass(
		"hclStandardLinkConstraintSet",
		&hclConstraintSetClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclStandardLinkConstraintSetClass_Members),
		HK_COUNT_OF(hclStandardLinkConstraintSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclStretchLinkConstraintSet_LinkClass_Members[] =
	{
		{ "particleA", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particleB", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "restLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stiffness", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclStretchLinkConstraintSetLinkClass(
		"hclStretchLinkConstraintSetLink",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclStretchLinkConstraintSet_LinkClass_Members),
		HK_COUNT_OF(hclStretchLinkConstraintSet_LinkClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclStretchLinkConstraintSetClass_Members[] =
	{
		{ "links", &hclStretchLinkConstraintSetLinkClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hclStretchLinkConstraintSetClass(
		"hclStretchLinkConstraintSet",
		&hclConstraintSetClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclStretchLinkConstraintSetClass_Members),
		HK_COUNT_OF(hclStretchLinkConstraintSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclTransitionConstraintSet_PerParticleClass_Members[] =
	{
		{ "particleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "referenceVertex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particleDelay", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclTransitionConstraintSetPerParticleClass(
		"hclTransitionConstraintSetPerParticle",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclTransitionConstraintSet_PerParticleClass_Members),
		HK_COUNT_OF(hclTransitionConstraintSet_PerParticleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclTransitionConstraintSetClass_Members[] =
	{
		{ "perParticleData", &hclTransitionConstraintSetPerParticleClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "transitionPeriod", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transitionPlusDelayPeriod", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "referenceMeshBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclTransitionConstraintSetClass(
		"hclTransitionConstraintSet",
		&hclConstraintSetClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclTransitionConstraintSetClass_Members),
		HK_COUNT_OF(hclTransitionConstraintSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclVolumeConstraint_FrameDataClass_Members[] =
	{
		{ "frameVector", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "weight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclVolumeConstraintFrameDataClass(
		"hclVolumeConstraintFrameData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclVolumeConstraint_FrameDataClass_Members),
		HK_COUNT_OF(hclVolumeConstraint_FrameDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclVolumeConstraint_ApplyDataClass_Members[] =
	{
		{ "frameVector", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stiffness", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclVolumeConstraintApplyDataClass(
		"hclVolumeConstraintApplyData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclVolumeConstraint_ApplyDataClass_Members),
		HK_COUNT_OF(hclVolumeConstraint_ApplyDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclVolumeConstraintClass_Members[] =
	{
		{ "frameDatas", &hclVolumeConstraintFrameDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "applyDatas", &hclVolumeConstraintApplyDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hclVolumeConstraintClass(
		"hclVolumeConstraint",
		&hclConstraintSetClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclVolumeConstraintClass_Members),
		HK_COUNT_OF(hclVolumeConstraintClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclClothContainerClass_Members[] =
	{
		{ "collidables", &hclCollidableClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "clothDatas", &hclClothDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hclClothContainerClass(
		"hclClothContainer",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclClothContainerClass_Members),
		HK_COUNT_OF(hclClothContainerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclOperatorClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hclOperatorClass(
		"hclOperator",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclOperatorClass_Members),
		HK_COUNT_OF(hclOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclBlendSomeVerticesOperator_BlendEntryClass_Members[] =
	{
		{ "vertexIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "blendWeight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclBlendSomeVerticesOperatorBlendEntryClass(
		"hclBlendSomeVerticesOperatorBlendEntry",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclBlendSomeVerticesOperator_BlendEntryClass_Members),
		HK_COUNT_OF(hclBlendSomeVerticesOperator_BlendEntryClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclBlendSomeVerticesOperatorClass_Members[] =
	{
		{ "blendEntries", &hclBlendSomeVerticesOperatorBlendEntryClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "bufferIdx_A", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bufferIdx_B", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bufferIdx_C", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclBlendSomeVerticesOperatorClass(
		"hclBlendSomeVerticesOperator",
		&hclOperatorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclBlendSomeVerticesOperatorClass_Members),
		HK_COUNT_OF(hclBlendSomeVerticesOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hclMeshMeshDeformOperatorScaleNormalBehaviourEnumItems[] =
	{
		{0, "SCALE_NORMAL_IGNORE"},
		{1, "SCALE_NORMAL_APPLY"},
		{2, "SCALE_NORMAL_INVERT"},
	};
	static const hkInternalClassEnum hclMeshMeshDeformOperatorEnums[] = {
		{"ScaleNormalBehaviour", hclMeshMeshDeformOperatorScaleNormalBehaviourEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hclMeshMeshDeformOperatorScaleNormalBehaviourEnum = reinterpret_cast<const hkClassEnum*>(&hclMeshMeshDeformOperatorEnums[0]);
	static hkInternalClassMember hclMeshMeshDeformOperator_TriangleVertexPairClass_Members[] =
	{
		{ "localPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "localNormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "triangleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "weight", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hclMeshMeshDeformOperatorTriangleVertexPair_DefaultStruct
		{
			int s_defaultOffsets[4];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_localNormal;
		};
		const hclMeshMeshDeformOperatorTriangleVertexPair_DefaultStruct hclMeshMeshDeformOperatorTriangleVertexPair_Default =
		{
			{-1,HK_OFFSET_OF(hclMeshMeshDeformOperatorTriangleVertexPair_DefaultStruct,m_localNormal),-1,-1},
			{0.0f,0.0f,1.0f}
		};
	}
	hkClass hclMeshMeshDeformOperatorTriangleVertexPairClass(
		"hclMeshMeshDeformOperatorTriangleVertexPair",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclMeshMeshDeformOperator_TriangleVertexPairClass_Members),
		HK_COUNT_OF(hclMeshMeshDeformOperator_TriangleVertexPairClass_Members),
		&hclMeshMeshDeformOperatorTriangleVertexPair_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclMeshMeshDeformOperatorClass_Members[] =
	{
		{ "inputBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "outputBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "inputTrianglesSubset", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "triangleVertexPairs", &hclMeshMeshDeformOperatorTriangleVertexPairClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "triangleVertexStartForVertex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "scaleNormalBehaviour", HK_NULL, hclMeshMeshDeformOperatorScaleNormalBehaviourEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "deformNormals", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclMeshMeshDeformOperatorClass(
		"hclMeshMeshDeformOperator",
		&hclOperatorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hclMeshMeshDeformOperatorEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hclMeshMeshDeformOperatorClass_Members),
		HK_COUNT_OF(hclMeshMeshDeformOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hclMoveParticlesOperatorForceUpgrade610EnumItems[] =
	{
		{0, "HCL_FORCE_UPGRADE610"},
	};
	static const hkInternalClassEnum hclMoveParticlesOperatorEnums[] = {
		{"ForceUpgrade610", hclMoveParticlesOperatorForceUpgrade610EnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hclMoveParticlesOperatorForceUpgrade610Enum = reinterpret_cast<const hkClassEnum*>(&hclMoveParticlesOperatorEnums[0]);
	static hkInternalClassMember hclMoveParticlesOperator_VertexParticlePairClass_Members[] =
	{
		{ "vertexIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclMoveParticlesOperatorVertexParticlePairClass(
		"hclMoveParticlesOperatorVertexParticlePair",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclMoveParticlesOperator_VertexParticlePairClass_Members),
		HK_COUNT_OF(hclMoveParticlesOperator_VertexParticlePairClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclMoveParticlesOperatorClass_Members[] =
	{
		{ "vertexParticlePairs", &hclMoveParticlesOperatorVertexParticlePairClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "simClothIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "refBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclMoveParticlesOperatorClass(
		"hclMoveParticlesOperator",
		&hclOperatorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hclMoveParticlesOperatorEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hclMoveParticlesOperatorClass_Members),
		HK_COUNT_OF(hclMoveParticlesOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclRecalculateAllNormalsOperatorClass_Members[] =
	{
		{ "bufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclRecalculateAllNormalsOperatorClass(
		"hclRecalculateAllNormalsOperator",
		&hclOperatorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclRecalculateAllNormalsOperatorClass_Members),
		HK_COUNT_OF(hclRecalculateAllNormalsOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclRecalculateSomeNormalsOperatorClass_Members[] =
	{
		{ "bufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vertexIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "triangleIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL }
	};
	hkClass hclRecalculateSomeNormalsOperatorClass(
		"hclRecalculateSomeNormalsOperator",
		&hclOperatorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclRecalculateSomeNormalsOperatorClass_Members),
		HK_COUNT_OF(hclRecalculateSomeNormalsOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclSimulateOperatorClass_Members[] =
	{
		{ "simClothIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "subSteps", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numberOfSolveIterations", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "constraintExecution", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL }
	};
	hkClass hclSimulateOperatorClass(
		"hclSimulateOperator",
		&hclOperatorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclSimulateOperatorClass_Members),
		HK_COUNT_OF(hclSimulateOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclSkinOperator_BoneInfluenceClass_Members[] =
	{
		{ "boneIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "weight", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclSkinOperatorBoneInfluenceClass(
		"hclSkinOperatorBoneInfluence",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclSkinOperator_BoneInfluenceClass_Members),
		HK_COUNT_OF(hclSkinOperator_BoneInfluenceClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclSkinOperatorClass_Members[] =
	{
		{ "skinPositions", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "skinNormals", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "skinTangents", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "skinBiTangents", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "inputBufferIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "outputBufferIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transformSetIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boneInfluences", &hclSkinOperatorBoneInfluenceClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "boneInfluenceStartPerVertex", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "boneFromSkinMeshTransforms", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_MATRIX4, 0, 0, 0, HK_NULL }
	};
	hkClass hclSkinOperatorClass(
		"hclSkinOperator",
		&hclOperatorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclSkinOperatorClass_Members),
		HK_COUNT_OF(hclSkinOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclCopyVerticesOperatorClass_Members[] =
	{
		{ "inputBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "outputBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numberOfVertices", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "copyNormals", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "startVertexIn", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "startVertexOut", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclCopyVerticesOperatorClass(
		"hclCopyVerticesOperator",
		&hclOperatorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclCopyVerticesOperatorClass_Members),
		HK_COUNT_OF(hclCopyVerticesOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclGatherAllVerticesOperatorClass_Members[] =
	{
		{ "vertexInputFromVertexOutput", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "inputBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "outputBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "gatherNormals", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclGatherAllVerticesOperatorClass(
		"hclGatherAllVerticesOperator",
		&hclOperatorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclGatherAllVerticesOperatorClass_Members),
		HK_COUNT_OF(hclGatherAllVerticesOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclGatherSomeVerticesOperator_VertexPairClass_Members[] =
	{
		{ "indexInput", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "indexOutput", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclGatherSomeVerticesOperatorVertexPairClass(
		"hclGatherSomeVerticesOperatorVertexPair",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclGatherSomeVerticesOperator_VertexPairClass_Members),
		HK_COUNT_OF(hclGatherSomeVerticesOperator_VertexPairClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclGatherSomeVerticesOperatorClass_Members[] =
	{
		{ "vertexPairs", &hclGatherSomeVerticesOperatorVertexPairClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "gatherNormals", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "inputBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "outputBufferIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclGatherSomeVerticesOperatorClass(
		"hclGatherSomeVerticesOperator",
		&hclOperatorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclGatherSomeVerticesOperatorClass_Members),
		HK_COUNT_OF(hclGatherSomeVerticesOperatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclSimClothData_SimulationInfoClass_Members[] =
	{
		{ "gravity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "globalDampingPerSecond", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "doNormals", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclSimClothDataSimulationInfoClass(
		"hclSimClothDataSimulationInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclSimClothData_SimulationInfoClass_Members),
		HK_COUNT_OF(hclSimClothData_SimulationInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclSimClothData_ParticleDataClass_Members[] =
	{
		{ "mass", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "invMass", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "friction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclSimClothDataParticleDataClass(
		"hclSimClothDataParticleData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclSimClothData_ParticleDataClass_Members),
		HK_COUNT_OF(hclSimClothData_ParticleDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclSimClothData_CollisionPairClass_Members[] =
	{
		{ "particleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collidableIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclSimClothDataCollisionPairClass(
		"hclSimClothDataCollisionPair",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclSimClothData_CollisionPairClass_Members),
		HK_COUNT_OF(hclSimClothData_CollisionPairClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclSimClothDataClass_Members[] =
	{
		{ "simulationInfo", &hclSimClothDataSimulationInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "particleDatas", &hclSimClothDataParticleDataClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "triangleIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "totalMass", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "perInstanceCollidables", &hclCollidableClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "staticConstraintSets", &hclConstraintSetClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "simClothPoses", &hclSimClothPoseClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "actions", &hclActionClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "enableMidPhase", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "staticCollisionPairs", &hclSimClothDataCollisionPairClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hclSimClothDataClass(
		"hclSimClothData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclSimClothDataClass_Members),
		HK_COUNT_OF(hclSimClothDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclSimClothPoseClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "positions", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL }
	};
	hkClass hclSimClothPoseClass(
		"hclSimClothPose",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclSimClothPoseClass_Members),
		HK_COUNT_OF(hclSimClothPoseClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclClothState_BufferAccessClass_Members[] =
	{
		{ "bufferIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "accessFlags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclClothStateBufferAccessClass(
		"hclClothStateBufferAccess",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclClothState_BufferAccessClass_Members),
		HK_COUNT_OF(hclClothState_BufferAccessClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclClothStateClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "operators", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "usedBuffers", &hclClothStateBufferAccessClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "usedTransformSets", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "usedSimCloths", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL }
	};
	hkClass hclClothStateClass(
		"hclClothState",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclClothStateClass_Members),
		HK_COUNT_OF(hclClothStateClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclTransformSetDefinitionClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numTransforms", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclTransformSetDefinitionClass(
		"hclTransformSetDefinition",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclTransformSetDefinitionClass_Members),
		HK_COUNT_OF(hclTransformSetDefinitionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hclToolNamedObjectReferenceClass_Members[] =
	{
		{ "pluginName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "objectName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hash", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclToolNamedObjectReferenceClass(
		"hclToolNamedObjectReference",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hclToolNamedObjectReferenceClass_Members),
		HK_COUNT_OF(hclToolNamedObjectReferenceClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hclTriangleSelectionInputTriangleSelectionTypeEnumItems[] =
	{
		{0, "TRIANGLE_SELECTION_ALL"},
		{1, "TRIANGLE_SELECTION_NONE"},
		{2, "TRIANGLE_SELECTION_CHANNEL"},
		{3, "TRIANGLE_SELECTION_INVERSE_CHANNEL"},
	};
	static const hkInternalClassEnum hclTriangleSelectionInputEnums[] = {
		{"TriangleSelectionType", hclTriangleSelectionInputTriangleSelectionTypeEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hclTriangleSelectionInputTriangleSelectionTypeEnum = reinterpret_cast<const hkClassEnum*>(&hclTriangleSelectionInputEnums[0]);
	static hkInternalClassMember hclTriangleSelectionInputClass_Members[] =
	{
		{ "type", HK_NULL, hclTriangleSelectionInputTriangleSelectionTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "channelName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclTriangleSelectionInputClass(
		"hclTriangleSelectionInput",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hclTriangleSelectionInputEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hclTriangleSelectionInputClass_Members),
		HK_COUNT_OF(hclTriangleSelectionInputClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hclVertexFloatInputVertexFloatTypeEnumItems[] =
	{
		{0, "VERTEX_FLOAT_CONSTANT"},
		{1, "VERTEX_FLOAT_CHANNEL"},
	};
	static const hkInternalClassEnum hclVertexFloatInputEnums[] = {
		{"VertexFloatType", hclVertexFloatInputVertexFloatTypeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hclVertexFloatInputVertexFloatTypeEnum = reinterpret_cast<const hkClassEnum*>(&hclVertexFloatInputEnums[0]);
	static hkInternalClassMember hclVertexFloatInputClass_Members[] =
	{
		{ "type", HK_NULL, hclVertexFloatInputVertexFloatTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "constantValue", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "channelName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclVertexFloatInputClass(
		"hclVertexFloatInput",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hclVertexFloatInputEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hclVertexFloatInputClass_Members),
		HK_COUNT_OF(hclVertexFloatInputClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hclVertexSelectionInputVertexSelectionTypeEnumItems[] =
	{
		{0, "VERTEX_SELECTION_ALL"},
		{1, "VERTEX_SELECTION_NONE"},
		{2, "VERTEX_SELECTION_CHANNEL"},
		{3, "VERTEX_SELECTION_INVERSE_CHANNEL"},
	};
	static const hkInternalClassEnum hclVertexSelectionInputEnums[] = {
		{"VertexSelectionType", hclVertexSelectionInputVertexSelectionTypeEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hclVertexSelectionInputVertexSelectionTypeEnum = reinterpret_cast<const hkClassEnum*>(&hclVertexSelectionInputEnums[0]);
	static hkInternalClassMember hclVertexSelectionInputClass_Members[] =
	{
		{ "type", HK_NULL, hclVertexSelectionInputVertexSelectionTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "channelName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hclVertexSelectionInputClass(
		"hclVertexSelectionInput",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hclVertexSelectionInputEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hclVertexSelectionInputClass_Members),
		HK_COUNT_OF(hclVertexSelectionInputClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkBitFieldClass_Members[] =
	{
		{ "words", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "numBits", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkBitFieldClass(
		"hkBitField",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkBitFieldClass_Members),
		HK_COUNT_OF(hkBitFieldClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkMultiThreadCheckAccessTypeEnumItems[] =
	{
		{0, "HK_ACCESS_IGNORE"},
		{1, "HK_ACCESS_RO"},
		{2, "HK_ACCESS_RW"},
	};
	static const hkInternalClassEnumItem hkMultiThreadCheckReadModeEnumItems[] =
	{
		{0, "THIS_OBJECT_ONLY"},
		{1, "RECURSIVE"},
	};
	static const hkInternalClassEnum hkMultiThreadCheckEnums[] = {
		{"AccessType", hkMultiThreadCheckAccessTypeEnumItems, 3, HK_NULL, 0 },
		{"ReadMode", hkMultiThreadCheckReadModeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkMultiThreadCheckAccessTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkMultiThreadCheckEnums[0]);
	const hkClassEnum* hkMultiThreadCheckReadModeEnum = reinterpret_cast<const hkClassEnum*>(&hkMultiThreadCheckEnums[1]);
	static hkInternalClassMember hkMultiThreadCheckClass_Members[] =
	{
		{ "threadId", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "markCount", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "markBitStack", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkMultiThreadCheckClass(
		"hkMultiThreadCheck",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkMultiThreadCheckEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkMultiThreadCheckClass_Members),
		HK_COUNT_OF(hkMultiThreadCheckClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkSweptTransformClass_Members[] =
	{
		{ "centerOfMass0", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "centerOfMass1", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotation0", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotation1", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "centerOfMassLocal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkSweptTransformClass(
		"hkSweptTransform",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkSweptTransformClass_Members),
		HK_COUNT_OF(hkSweptTransformClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkMonitorStreamStringMap_StringMapClass_Members[] =
	{
		{ "id", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT64, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_8, 0, HK_NULL },
		{ "string", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkMonitorStreamStringMapStringMapClass(
		"hkMonitorStreamStringMapStringMap",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkMonitorStreamStringMap_StringMapClass_Members),
		HK_COUNT_OF(hkMonitorStreamStringMap_StringMapClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkMonitorStreamStringMapClass_Members[] =
	{
		{ "map", &hkMonitorStreamStringMapStringMapClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkMonitorStreamStringMapClass(
		"hkMonitorStreamStringMap",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkMonitorStreamStringMapClass_Members),
		HK_COUNT_OF(hkMonitorStreamStringMapClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkMonitorStreamFrameInfoAbsoluteTimeCounterEnumItems[] =
	{
		{0, "ABSOLUTE_TIME_TIMER_0"},
		{1, "ABSOLUTE_TIME_TIMER_1"},
		{-1, "ABSOLUTE_TIME_NOT_TIMED"},
	};
	static const hkInternalClassEnum hkMonitorStreamFrameInfoEnums[] = {
		{"AbsoluteTimeCounter", hkMonitorStreamFrameInfoAbsoluteTimeCounterEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkMonitorStreamFrameInfoAbsoluteTimeCounterEnum = reinterpret_cast<const hkClassEnum*>(&hkMonitorStreamFrameInfoEnums[0]);
	static hkInternalClassMember hkMonitorStreamFrameInfoClass_Members[] =
	{
		{ "heading", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "indexOfTimer0", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "indexOfTimer1", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "absoluteTimeCounter", HK_NULL, hkMonitorStreamFrameInfoAbsoluteTimeCounterEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "timerFactor0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "timerFactor1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "threadId", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "frameStreamStart", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "frameStreamEnd", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkMonitorStreamFrameInfoClass(
		"hkMonitorStreamFrameInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkMonitorStreamFrameInfoEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkMonitorStreamFrameInfoClass_Members),
		HK_COUNT_OF(hkMonitorStreamFrameInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkMonitorStreamColorTable_ColorPairClass_Members[] =
	{
		{ "colorName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "color", HK_NULL, hkColorExtendedColorsEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL }
	};
	hkClass hkMonitorStreamColorTableColorPairClass(
		"hkMonitorStreamColorTableColorPair",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkMonitorStreamColorTable_ColorPairClass_Members),
		HK_COUNT_OF(hkMonitorStreamColorTable_ColorPairClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkMonitorStreamColorTableClass_Members[] =
	{
		{ "colorPairs", &hkMonitorStreamColorTableColorPairClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "defaultColor", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkMonitorStreamColorTableClass(
		"hkMonitorStreamColorTable",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkMonitorStreamColorTableClass_Members),
		HK_COUNT_OF(hkMonitorStreamColorTableClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkBaseObjectClass(
		"hkBaseObject",
		HK_NULL,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkReferencedObjectClass_Members[] =
	{
		{ "memSizeAndFlags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "referenceCount", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkReferencedObjectClass(
		"hkReferencedObject",
		&hkBaseObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkReferencedObjectClass_Members),
		HK_COUNT_OF(hkReferencedObjectClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkClassSignatureFlagsEnumItems[] =
	{
		{1, "SIGNATURE_LOCAL"},
	};
	static const hkInternalClassEnumItem hkClassFlagValuesEnumItems[] =
	{
		{0, "FLAGS_NONE"},
		{1, "FLAGS_NOT_SERIALIZABLE"},
	};
	static const hkInternalClassEnum hkClassEnums[] = {
		{"SignatureFlags", hkClassSignatureFlagsEnumItems, 1, HK_NULL, 0 },
		{"FlagValues", hkClassFlagValuesEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkClassSignatureFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkClassEnums[0]);
	const hkClassEnum* hkClassFlagValuesEnum = reinterpret_cast<const hkClassEnum*>(&hkClassEnums[1]);
	static hkInternalClassMember hkClassClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "parent", &hkClassClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "objectSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numImplementedInterfaces", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "declaredEnums", &hkClassEnumClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "declaredMembers", &hkClassMemberClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "defaults", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "attributes", &hkCustomAttributesClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "flags", HK_NULL, hkClassFlagValuesEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "describedVersion", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkClassClass(
		"hkClass",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkClassEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkClassClass_Members),
		HK_COUNT_OF(hkClassClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkClassEnumFlagValuesEnumItems[] =
	{
		{0, "FLAGS_NONE"},
	};
	static const hkInternalClassEnum hkClassEnumEnums[] = {
		{"FlagValues", hkClassEnumFlagValuesEnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hkClassEnumFlagValuesEnum = reinterpret_cast<const hkClassEnum*>(&hkClassEnumEnums[0]);
	static hkInternalClassMember hkClassEnum_ItemClass_Members[] =
	{
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkClassEnumItemClass(
		"hkClassEnumItem",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkClassEnum_ItemClass_Members),
		HK_COUNT_OF(hkClassEnum_ItemClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkClassEnumClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "items", &hkClassEnumItemClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "attributes", &hkCustomAttributesClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "flags", HK_NULL, hkClassEnumFlagValuesEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL }
	};
	hkClass hkClassEnumClass(
		"hkClassEnum",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkClassEnumEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkClassEnumClass_Members),
		HK_COUNT_OF(hkClassEnumClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkClassMemberTypeEnumItems[] =
	{
		{0, "TYPE_VOID"},
		{1, "TYPE_BOOL"},
		{2, "TYPE_CHAR"},
		{3, "TYPE_INT8"},
		{4, "TYPE_UINT8"},
		{5, "TYPE_INT16"},
		{6, "TYPE_UINT16"},
		{7, "TYPE_INT32"},
		{8, "TYPE_UINT32"},
		{9, "TYPE_INT64"},
		{10, "TYPE_UINT64"},
		{11, "TYPE_REAL"},
		{12, "TYPE_VECTOR4"},
		{13, "TYPE_QUATERNION"},
		{14, "TYPE_MATRIX3"},
		{15, "TYPE_ROTATION"},
		{16, "TYPE_QSTRANSFORM"},
		{17, "TYPE_MATRIX4"},
		{18, "TYPE_TRANSFORM"},
		{19, "TYPE_ZERO"},
		{20, "TYPE_POINTER"},
		{21, "TYPE_FUNCTIONPOINTER"},
		{22, "TYPE_ARRAY"},
		{23, "TYPE_INPLACEARRAY"},
		{24, "TYPE_ENUM"},
		{25, "TYPE_STRUCT"},
		{26, "TYPE_SIMPLEARRAY"},
		{27, "TYPE_HOMOGENEOUSARRAY"},
		{28, "TYPE_VARIANT"},
		{29, "TYPE_CSTRING"},
		{30, "TYPE_ULONG"},
		{31, "TYPE_FLAGS"},
		{32, "TYPE_MAX"},
	};
	static const hkInternalClassEnumItem hkClassMemberFlagValuesEnumItems[] =
	{
		{0, "FLAGS_NONE"},
		{128, "ALIGN_8"},
		{256, "ALIGN_16"},
		{512, "NOT_OWNED"},
		{1024, "SERIALIZE_IGNORED"},
	};
	static const hkInternalClassEnumItem hkClassMemberDeprecatedFlagValuesEnumItems[] =
	{
		{8, "DEPRECATED_SIZE_8"},
		{8, "DEPRECATED_ENUM_8"},
		{16, "DEPRECATED_SIZE_16"},
		{16, "DEPRECATED_ENUM_16"},
		{32, "DEPRECATED_SIZE_32"},
		{32, "DEPRECATED_ENUM_32"},
	};
	static const hkInternalClassEnum hkClassMemberEnums[] = {
		{"Type", hkClassMemberTypeEnumItems, 33, HK_NULL, 0 },
		{"FlagValues", hkClassMemberFlagValuesEnumItems, 5, HK_NULL, 0 },
		{"DeprecatedFlagValues", hkClassMemberDeprecatedFlagValuesEnumItems, 6, HK_NULL, 0 }
	};
	const hkClassEnum* hkClassMemberTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkClassMemberEnums[0]);
	const hkClassEnum* hkClassMemberFlagValuesEnum = reinterpret_cast<const hkClassEnum*>(&hkClassMemberEnums[1]);
	const hkClassEnum* hkClassMemberDeprecatedFlagValuesEnum = reinterpret_cast<const hkClassEnum*>(&hkClassMemberEnums[2]);
	static hkInternalClassMember hkClassMemberClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "class", &hkClassClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "enum", &hkClassEnumClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, hkClassMemberTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "subtype", HK_NULL, hkClassMemberTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "cArraySize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "flags", HK_NULL, hkClassMemberFlagValuesEnum, hkClassMember::TYPE_FLAGS, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "offset", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "attributes", &hkCustomAttributesClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkClassMemberClass(
		"hkClassMember",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkClassMemberEnums),
		3,
		reinterpret_cast<const hkClassMember*>(hkClassMemberClass_Members),
		HK_COUNT_OF(hkClassMemberClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkCustomAttributes_AttributeClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_VARIANT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkCustomAttributesAttributeClass(
		"hkCustomAttributesAttribute",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkCustomAttributes_AttributeClass_Members),
		HK_COUNT_OF(hkCustomAttributes_AttributeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkCustomAttributesClass_Members[] =
	{
		{ "attributes", &hkCustomAttributesAttributeClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkCustomAttributesClass(
		"hkCustomAttributes",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkCustomAttributesClass_Members),
		HK_COUNT_OF(hkCustomAttributesClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkColorExtendedColorsEnumItems[] =
	{
		{4286578688u/*0xFF800000*/, "MAROON"},
		{4287299584u/*0xFF8B0000*/, "DARKRED"},
		{4294901760u/*0xFFFF0000*/, "RED"},
		{4294948545u/*0xFFFFB6C1*/, "LIGHTPINK"},
		{4292613180u/*0xFFDC143C*/, "CRIMSON"},
		{4292571283u/*0xFFDB7093*/, "PALEVIOLETRED"},
		{4294928820u/*0xFFFF69B4*/, "HOTPINK"},
		{4294907027u/*0xFFFF1493*/, "DEEPPINK"},
		{4291237253u/*0xFFC71585*/, "MEDIUMVIOLETRED"},
		{4286578816u/*0xFF800080*/, "PURPLE"},
		{4287299723u/*0xFF8B008B*/, "DARKMAGENTA"},
		{4292505814u/*0xFFDA70D6*/, "ORCHID"},
		{4292394968u/*0xFFD8BFD8*/, "THISTLE"},
		{4292714717u/*0xFFDDA0DD*/, "PLUM"},
		{4293821166u/*0xFFEE82EE*/, "VIOLET"},
		{4294902015u/*0xFFFF00FF*/, "FUCHSIA"},
		{4294902015u/*0xFFFF00FF*/, "MAGENTA"},
		{4290401747u/*0xFFBA55D3*/, "MEDIUMORCHID"},
		{4287889619u/*0xFF9400D3*/, "DARKVIOLET"},
		{4288230092u/*0xFF9932CC*/, "DARKORCHID"},
		{4287245282u/*0xFF8A2BE2*/, "BLUEVIOLET"},
		{4283105410u/*0xFF4B0082*/, "INDIGO"},
		{4287852763u/*0xFF9370DB*/, "MEDIUMPURPLE"},
		{4285160141u/*0xFF6A5ACD*/, "SLATEBLUE"},
		{4286277870u/*0xFF7B68EE*/, "MEDIUMSLATEBLUE"},
		{4278190219u/*0xFF00008B*/, "DARKBLUE"},
		{4278190285u/*0xFF0000CD*/, "MEDIUMBLUE"},
		{4278190335u/*0xFF0000FF*/, "BLUE"},
		{4278190208u/*0xFF000080*/, "NAVY"},
		{4279834992u/*0xFF191970*/, "MIDNIGHTBLUE"},
		{4282924427u/*0xFF483D8B*/, "DARKSLATEBLUE"},
		{4282477025u/*0xFF4169E1*/, "ROYALBLUE"},
		{4284782061u/*0xFF6495ED*/, "CORNFLOWERBLUE"},
		{4289774814u/*0xFFB0C4DE*/, "LIGHTSTEELBLUE"},
		{4293982463u/*0xFFF0F8FF*/, "ALICEBLUE"},
		{4294506751u/*0xFFF8F8FF*/, "GHOSTWHITE"},
		{4293322490u/*0xFFE6E6FA*/, "LAVENDER"},
		{4280193279u/*0xFF1E90FF*/, "DODGERBLUE"},
		{4282811060u/*0xFF4682B4*/, "STEELBLUE"},
		{4278239231u/*0xFF00BFFF*/, "DEEPSKYBLUE"},
		{4285563024u/*0xFF708090*/, "SLATEGRAY"},
		{4286023833u/*0xFF778899*/, "LIGHTSLATEGRAY"},
		{4287090426u/*0xFF87CEFA*/, "LIGHTSKYBLUE"},
		{4287090411u/*0xFF87CEEB*/, "SKYBLUE"},
		{4289583334u/*0xFFADD8E6*/, "LIGHTBLUE"},
		{4278222976u/*0xFF008080*/, "TEAL"},
		{4278225803u/*0xFF008B8B*/, "DARKCYAN"},
		{4278243025u/*0xFF00CED1*/, "DARKTURQUOISE"},
		{4278255615u/*0xFF00FFFF*/, "CYAN"},
		{4282962380u/*0xFF48D1CC*/, "MEDIUMTURQUOISE"},
		{4284456608u/*0xFF5F9EA0*/, "CADETBLUE"},
		{4289720046u/*0xFFAFEEEE*/, "PALETURQUOISE"},
		{4292935679u/*0xFFE0FFFF*/, "LIGHTCYAN"},
		{4293984255u/*0xFFF0FFFF*/, "AZURE"},
		{4280332970u/*0xFF20B2AA*/, "LIGHTSEAGREEN"},
		{4282441936u/*0xFF40E0D0*/, "TURQUOISE"},
		{4289781990u/*0xFFB0E0E6*/, "POWDERBLUE"},
		{4281290575u/*0xFF2F4F4F*/, "DARKSLATEGRAY"},
		{4286578644u/*0xFF7FFFD4*/, "AQUAMARINE"},
		{4278254234u/*0xFF00FA9A*/, "MEDIUMSPRINGGREEN"},
		{4284927402u/*0xFF66CDAA*/, "MEDIUMAQUAMARINE"},
		{4278255487u/*0xFF00FF7F*/, "SPRINGGREEN"},
		{4282168177u/*0xFF3CB371*/, "MEDIUMSEAGREEN"},
		{4281240407u/*0xFF2E8B57*/, "SEAGREEN"},
		{4281519410u/*0xFF32CD32*/, "LIMEGREEN"},
		{4278215680u/*0xFF006400*/, "DARKGREEN"},
		{4278222848u/*0xFF008000*/, "GREEN"},
		{4278255360u/*0xFF00FF00*/, "LIME"},
		{4280453922u/*0xFF228B22*/, "FORESTGREEN"},
		{4287609999u/*0xFF8FBC8F*/, "DARKSEAGREEN"},
		{4287688336u/*0xFF90EE90*/, "LIGHTGREEN"},
		{4288215960u/*0xFF98FB98*/, "PALEGREEN"},
		{4294311930u/*0xFFF5FFFA*/, "MINTCREAM"},
		{4293984240u/*0xFFF0FFF0*/, "HONEYDEW"},
		{4286578432u/*0xFF7FFF00*/, "CHARTREUSE"},
		{4286381056u/*0xFF7CFC00*/, "LAWNGREEN"},
		{4285238819u/*0xFF6B8E23*/, "OLIVEDRAB"},
		{4283788079u/*0xFF556B2F*/, "DARKOLIVEGREEN"},
		{4288335154u/*0xFF9ACD32*/, "YELLOWGREEN"},
		{4289593135u/*0xFFADFF2F*/, "GREENYELLOW"},
		{4294309340u/*0xFFF5F5DC*/, "BEIGE"},
		{4294635750u/*0xFFFAF0E6*/, "LINEN"},
		{4294638290u/*0xFFFAFAD2*/, "LIGHTGOLDENRODYELLOW"},
		{4286611456u/*0xFF808000*/, "OLIVE"},
		{4294967040u/*0xFFFFFF00*/, "YELLOW"},
		{4294967264u/*0xFFFFFFE0*/, "LIGHTYELLOW"},
		{4294967280u/*0xFFFFFFF0*/, "IVORY"},
		{4290623339u/*0xFFBDB76B*/, "DARKKHAKI"},
		{4293977740u/*0xFFF0E68C*/, "KHAKI"},
		{4293847210u/*0xFFEEE8AA*/, "PALEGOLDENROD"},
		{4294303411u/*0xFFF5DEB3*/, "WHEAT"},
		{4294956800u/*0xFFFFD700*/, "GOLD"},
		{4294965965u/*0xFFFFFACD*/, "LEMONCHIFFON"},
		{4294963157u/*0xFFFFEFD5*/, "PAPAYAWHIP"},
		{4290283019u/*0xFFB8860B*/, "DARKGOLDENROD"},
		{4292519200u/*0xFFDAA520*/, "GOLDENROD"},
		{4294634455u/*0xFFFAEBD7*/, "ANTIQUEWHITE"},
		{4294965468u/*0xFFFFF8DC*/, "CORNSILK"},
		{4294833638u/*0xFFFDF5E6*/, "OLDLACE"},
		{4294960309u/*0xFFFFE4B5*/, "MOCCASIN"},
		{4294958765u/*0xFFFFDEAD*/, "NAVAJOWHITE"},
		{4294944000u/*0xFFFFA500*/, "ORANGE"},
		{4294960324u/*0xFFFFE4C4*/, "BISQUE"},
		{4291998860u/*0xFFD2B48C*/, "TAN"},
		{4294937600u/*0xFFFF8C00*/, "DARKORANGE"},
		{4292786311u/*0xFFDEB887*/, "BURLYWOOD"},
		{4287317267u/*0xFF8B4513*/, "SADDLEBROWN"},
		{4294222944u/*0xFFF4A460*/, "SANDYBROWN"},
		{4294962125u/*0xFFFFEBCD*/, "BLANCHEDALMOND"},
		{4294963445u/*0xFFFFF0F5*/, "LAVENDERBLUSH"},
		{4294964718u/*0xFFFFF5EE*/, "SEASHELL"},
		{4294966000u/*0xFFFFFAF0*/, "FLORALWHITE"},
		{4294966010u/*0xFFFFFAFA*/, "SNOW"},
		{4291659071u/*0xFFCD853F*/, "PERU"},
		{4294957753u/*0xFFFFDAB9*/, "PEACHPUFF"},
		{4291979550u/*0xFFD2691E*/, "CHOCOLATE"},
		{4288696877u/*0xFFA0522D*/, "SIENNA"},
		{4294942842u/*0xFFFFA07A*/, "LIGHTSALMON"},
		{4294934352u/*0xFFFF7F50*/, "CORAL"},
		{4293498490u/*0xFFE9967A*/, "DARKSALMON"},
		{4294960353u/*0xFFFFE4E1*/, "MISTYROSE"},
		{4294919424u/*0xFFFF4500*/, "ORANGERED"},
		{4294606962u/*0xFFFA8072*/, "SALMON"},
		{4294927175u/*0xFFFF6347*/, "TOMATO"},
		{4290547599u/*0xFFBC8F8F*/, "ROSYBROWN"},
		{4294951115u/*0xFFFFC0CB*/, "PINK"},
		{4291648604u/*0xFFCD5C5C*/, "INDIANRED"},
		{4293951616u/*0xFFF08080*/, "LIGHTCORAL"},
		{4289014314u/*0xFFA52A2A*/, "BROWN"},
		{4289864226u/*0xFFB22222*/, "FIREBRICK"},
		{4278190080u/*0xFF000000*/, "BLACK"},
		{4285098345u/*0xFF696969*/, "DIMGRAY"},
		{4286611584u/*0xFF808080*/, "GRAY"},
		{4289309097u/*0xFFA9A9A9*/, "DARKGRAY"},
		{4290822336u/*0xFFC0C0C0*/, "SILVER"},
		{4292072403u/*0xFFD3D3D3*/, "LIGHTGREY"},
		{4292664540u/*0xFFDCDCDC*/, "GAINSBORO"},
		{4294309365u/*0xFFF5F5F5*/, "WHITESMOKE"},
		{4294967295u/*0xFFFFFFFF*/, "WHITE"},
		{4287137928u/*0xff888888*/, "GREY"},
		{4282400832u/*0xff404040*/, "GREY25"},
		{4286611584u/*0xff808080*/, "GREY50"},
		{4290822336u/*0xffc0c0c0*/, "GREY75"},
	};
	static const hkInternalClassEnum hkColorEnums[] = {
		{"ExtendedColors", hkColorExtendedColorsEnumItems, 143, HK_NULL, 0 }
	};
	const hkClassEnum* hkColorExtendedColorsEnum = reinterpret_cast<const hkClassEnum*>(&hkColorEnums[0]);
	hkClass hkColorClass(
		"hkColor",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkColorEnums),
		1,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkAabbClass_Members[] =
	{
		{ "min", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "max", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkAabbClass(
		"hkAabb",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkAabbClass_Members),
		HK_COUNT_OF(hkAabbClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkAabbUint32Class_Members[] =
	{
		{ "min", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 3, 0|hkClassMember::ALIGN_16, 0, HK_NULL },
		{ "expansionMin", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL },
		{ "expansionShift", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "max", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL },
		{ "expansionMax", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL },
		{ "shapeKeyByte", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkAabbUint32Class(
		"hkAabbUint32",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkAabbUint32Class_Members),
		HK_COUNT_OF(hkAabbUint32Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkLocalFrameClass(
		"hkLocalFrame",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkSimpleLocalFrameClass_Members[] =
	{
		{ "transform", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "children", &hkLocalFrameClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "parentFrame", &hkLocalFrameClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkSimpleLocalFrameClass(
		"hkSimpleLocalFrame",
		&hkLocalFrameClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkSimpleLocalFrameClass_Members),
		HK_COUNT_OF(hkSimpleLocalFrameClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkSphereClass_Members[] =
	{
		{ "pos", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkSphereClass(
		"hkSphere",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkSphereClass_Members),
		HK_COUNT_OF(hkSphereClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkContactPointClass_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "separatingNormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkContactPointClass(
		"hkContactPoint",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkContactPointClass_Members),
		HK_COUNT_OF(hkContactPointClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkContactPointMaterialFlagEnumEnumItems[] =
	{
		{1, "CONTACT_IS_NEW_AND_POTENTIAL"},
		{2, "CONTACT_USES_SOLVER_PATH2"},
		{4, "CONTACT_BREAKOFF_OBJECT_ID"},
	};
	static const hkInternalClassEnum hkContactPointMaterialEnums[] = {
		{"FlagEnum", hkContactPointMaterialFlagEnumEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkContactPointMaterialFlagEnumEnum = reinterpret_cast<const hkClassEnum*>(&hkContactPointMaterialEnums[0]);
	static hkInternalClassMember hkContactPointMaterialClass_Members[] =
	{
		{ "userData", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "friction", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "restitution", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxImpulse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "flags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkContactPointMaterialClass(
		"hkContactPointMaterial",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkContactPointMaterialEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkContactPointMaterialClass_Members),
		HK_COUNT_OF(hkContactPointMaterialClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkMotionStateClass_Members[] =
	{
		{ "transform", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sweptTransform", &hkSweptTransformClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deltaAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "objectRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "linearDamping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angularDamping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxLinearVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxAngularVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deactivationClass", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkMotionStateClass(
		"hkMotionState",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkMotionStateClass_Members),
		HK_COUNT_OF(hkMotionStateClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxAnimatedFloatClass_Members[] =
	{
		{ "floats", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "hint", HK_NULL, hkxAttributeHintEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkxAnimatedFloatClass(
		"hkxAnimatedFloat",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxAnimatedFloatClass_Members),
		HK_COUNT_OF(hkxAnimatedFloatClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxAnimatedMatrixClass_Members[] =
	{
		{ "matrices", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_MATRIX4, 0, 0, 0, HK_NULL },
		{ "hint", HK_NULL, hkxAttributeHintEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkxAnimatedMatrixClass(
		"hkxAnimatedMatrix",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxAnimatedMatrixClass_Members),
		HK_COUNT_OF(hkxAnimatedMatrixClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxAnimatedQuaternionClass_Members[] =
	{
		{ "quaternions", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_QUATERNION, 0, 0, 0, HK_NULL }
	};
	hkClass hkxAnimatedQuaternionClass(
		"hkxAnimatedQuaternion",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxAnimatedQuaternionClass_Members),
		HK_COUNT_OF(hkxAnimatedQuaternionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxAnimatedVectorClass_Members[] =
	{
		{ "vectors", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL },
		{ "hint", HK_NULL, hkxAttributeHintEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkxAnimatedVectorClass(
		"hkxAnimatedVector",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxAnimatedVectorClass_Members),
		HK_COUNT_OF(hkxAnimatedVectorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkxAttributeHintEnumItems[] =
	{
		{0, "HINT_NONE"},
		{1, "HINT_IGNORE"},
		{2, "HINT_TRANSFORM"},
		{4, "HINT_SCALE"},
		{6, "HINT_TRANSFORM_AND_SCALE"},
		{8, "HINT_FLIP"},
	};
	static const hkInternalClassEnum hkxAttributeEnums[] = {
		{"Hint", hkxAttributeHintEnumItems, 6, HK_NULL, 0 }
	};
	const hkClassEnum* hkxAttributeHintEnum = reinterpret_cast<const hkClassEnum*>(&hkxAttributeEnums[0]);
	static hkInternalClassMember hkxAttributeClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_VARIANT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxAttributeClass(
		"hkxAttribute",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkxAttributeEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkxAttributeClass_Members),
		HK_COUNT_OF(hkxAttributeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxAttributeGroupClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "attributes", &hkxAttributeClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkxAttributeGroupClass(
		"hkxAttributeGroup",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxAttributeGroupClass_Members),
		HK_COUNT_OF(hkxAttributeGroupClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxAttributeHolderClass_Members[] =
	{
		{ "attributeGroups", &hkxAttributeGroupClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkxAttributeHolderClass(
		"hkxAttributeHolder",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxAttributeHolderClass_Members),
		HK_COUNT_OF(hkxAttributeHolderClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxSparselyAnimatedBoolClass_Members[] =
	{
		{ "bools", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_BOOL, 0, 0, 0, HK_NULL },
		{ "times", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL }
	};
	hkClass hkxSparselyAnimatedBoolClass(
		"hkxSparselyAnimatedBool",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxSparselyAnimatedBoolClass_Members),
		HK_COUNT_OF(hkxSparselyAnimatedBoolClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxSparselyAnimatedEnumClass_Members[] =
	{
		{ "type", &hkClassEnumClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkxSparselyAnimatedEnumClass(
		"hkxSparselyAnimatedEnum",
		&hkxSparselyAnimatedIntClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxSparselyAnimatedEnumClass_Members),
		HK_COUNT_OF(hkxSparselyAnimatedEnumClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxSparselyAnimatedIntClass_Members[] =
	{
		{ "ints", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "times", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL }
	};
	hkClass hkxSparselyAnimatedIntClass(
		"hkxSparselyAnimatedInt",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxSparselyAnimatedIntClass_Members),
		HK_COUNT_OF(hkxSparselyAnimatedIntClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxSparselyAnimatedString_StringTypeClass_Members[] =
	{
		{ "string", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxSparselyAnimatedStringStringTypeClass(
		"hkxSparselyAnimatedStringStringType",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxSparselyAnimatedString_StringTypeClass_Members),
		HK_COUNT_OF(hkxSparselyAnimatedString_StringTypeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxSparselyAnimatedStringClass_Members[] =
	{
		{ "strings", &hkxSparselyAnimatedStringStringTypeClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "times", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL }
	};
	hkClass hkxSparselyAnimatedStringClass(
		"hkxSparselyAnimatedString",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxSparselyAnimatedStringClass_Members),
		HK_COUNT_OF(hkxSparselyAnimatedStringClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxCameraClass_Members[] =
	{
		{ "from", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "focus", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "up", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fov", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "far", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "near", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "leftHanded", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxCameraClass(
		"hkxCamera",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxCameraClass_Members),
		HK_COUNT_OF(hkxCameraClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxEnvironment_VariableClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "value", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxEnvironmentVariableClass(
		"hkxEnvironmentVariable",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxEnvironment_VariableClass_Members),
		HK_COUNT_OF(hkxEnvironment_VariableClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxEnvironmentClass_Members[] =
	{
		{ "variables", &hkxEnvironmentVariableClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkxEnvironmentClass(
		"hkxEnvironment",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxEnvironmentClass_Members),
		HK_COUNT_OF(hkxEnvironmentClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxNode_AnnotationDataClass_Members[] =
	{
		{ "time", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "description", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxNodeAnnotationDataClass(
		"hkxNodeAnnotationData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxNode_AnnotationDataClass_Members),
		HK_COUNT_OF(hkxNode_AnnotationDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxNodeClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "object", HK_NULL, HK_NULL, hkClassMember::TYPE_VARIANT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "keyFrames", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_MATRIX4, 0, 0, 0, HK_NULL },
		{ "children", &hkxNodeClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "annotations", &hkxNodeAnnotationDataClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "userProperties", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "selected", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxNodeClass(
		"hkxNode",
		&hkxAttributeHolderClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxNodeClass_Members),
		HK_COUNT_OF(hkxNodeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkxLightLightTypeEnumItems[] =
	{
		{0, "POINT_LIGHT"},
		{1, "DIRECTIONAL_LIGHT"},
		{2, "SPOT_LIGHT"},
	};
	static const hkInternalClassEnum hkxLightEnums[] = {
		{"LightType", hkxLightLightTypeEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkxLightLightTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkxLightEnums[0]);
	static hkInternalClassMember hkxLightClass_Members[] =
	{
		{ "type", HK_NULL, hkxLightLightTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "direction", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "color", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxLightClass(
		"hkxLight",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkxLightEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkxLightClass_Members),
		HK_COUNT_OF(hkxLightClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkxMaterialTextureTypeEnumItems[] =
	{
		{0, "TEX_UNKNOWN"},
		{1, "TEX_DIFFUSE"},
		{2, "TEX_REFLECTION"},
		{3, "TEX_BUMP"},
		{4, "TEX_NORMAL"},
		{5, "TEX_DISPLACEMENT"},
		{6, "TEX_SPECULAR"},
		{7, "TEX_SPECULARANDGLOSS"},
		{8, "TEX_OPACITY"},
		{9, "TEX_EMISSIVE"},
		{10, "TEX_REFRACTION"},
		{11, "TEX_GLOSS"},
		{12, "TEX_NOTEXPORTED"},
	};
	static const hkInternalClassEnum hkxMaterialEnums[] = {
		{"TextureType", hkxMaterialTextureTypeEnumItems, 13, HK_NULL, 0 }
	};
	const hkClassEnum* hkxMaterialTextureTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkxMaterialEnums[0]);
	static hkInternalClassMember hkxMaterial_TextureStageClass_Members[] =
	{
		{ "texture", HK_NULL, HK_NULL, hkClassMember::TYPE_VARIANT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "usageHint", HK_NULL, hkxMaterialTextureTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "tcoordChannel", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkxMaterialTextureStage_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkInt32 m_tcoordChannel;
		};
		const hkxMaterialTextureStage_DefaultStruct hkxMaterialTextureStage_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkxMaterialTextureStage_DefaultStruct,m_tcoordChannel)},
			-1
		};
	}
	hkClass hkxMaterialTextureStageClass(
		"hkxMaterialTextureStage",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxMaterial_TextureStageClass_Members),
		HK_COUNT_OF(hkxMaterial_TextureStageClass_Members),
		&hkxMaterialTextureStage_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxMaterialClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stages", &hkxMaterialTextureStageClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "diffuseColor", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ambientColor", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "specularColor", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "emissiveColor", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "subMaterials", &hkxMaterialClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "extraData", HK_NULL, HK_NULL, hkClassMember::TYPE_VARIANT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxMaterialClass(
		"hkxMaterial",
		&hkxAttributeHolderClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkxMaterialEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkxMaterialClass_Members),
		HK_COUNT_OF(hkxMaterialClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkxMaterialEffectEffectTypeEnumItems[] =
	{
		{0, "EFFECT_TYPE_INVALID"},
		{1, "EFFECT_TYPE_UNKNOWN"},
		{2, "EFFECT_TYPE_HLSL_FX_INLINE"},
		{3, "EFFECT_TYPE_CG_FX_INLINE"},
		{4, "EFFECT_TYPE_HLSL_FX_FILENAME"},
		{5, "EFFECT_TYPE_CG_FX_FILENAME"},
		{6, "EFFECT_TYPE_MAX_ID"},
	};
	static const hkInternalClassEnum hkxMaterialEffectEnums[] = {
		{"EffectType", hkxMaterialEffectEffectTypeEnumItems, 7, HK_NULL, 0 }
	};
	const hkClassEnum* hkxMaterialEffectEffectTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkxMaterialEffectEnums[0]);
	static hkInternalClassMember hkxMaterialEffectClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, hkxMaterialEffectEffectTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkxMaterialEffectClass(
		"hkxMaterialEffect",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkxMaterialEffectEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkxMaterialEffectClass_Members),
		HK_COUNT_OF(hkxMaterialEffectClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkxMaterialShaderShaderTypeEnumItems[] =
	{
		{0, "EFFECT_TYPE_INVALID"},
		{1, "EFFECT_TYPE_UNKNOWN"},
		{2, "EFFECT_TYPE_HLSL_INLINE"},
		{3, "EFFECT_TYPE_CG_INLINE"},
		{4, "EFFECT_TYPE_HLSL_FILENAME"},
		{5, "EFFECT_TYPE_CG_FILENAME"},
		{6, "EFFECT_TYPE_MAX_ID"},
	};
	static const hkInternalClassEnum hkxMaterialShaderEnums[] = {
		{"ShaderType", hkxMaterialShaderShaderTypeEnumItems, 7, HK_NULL, 0 }
	};
	const hkClassEnum* hkxMaterialShaderShaderTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkxMaterialShaderEnums[0]);
	static hkInternalClassMember hkxMaterialShaderClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, hkxMaterialShaderShaderTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "vertexEntryName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "geomEntryName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pixelEntryName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkxMaterialShaderClass(
		"hkxMaterialShader",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkxMaterialShaderEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkxMaterialShaderClass_Members),
		HK_COUNT_OF(hkxMaterialShaderClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxMaterialShaderSetClass_Members[] =
	{
		{ "shaders", &hkxMaterialShaderClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkxMaterialShaderSetClass(
		"hkxMaterialShaderSet",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxMaterialShaderSetClass_Members),
		HK_COUNT_OF(hkxMaterialShaderSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxTextureFileClass_Members[] =
	{
		{ "filename", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "originalFilename", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxTextureFileClass(
		"hkxTextureFile",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxTextureFileClass_Members),
		HK_COUNT_OF(hkxTextureFileClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxTextureInplaceClass_Members[] =
	{
		{ "fileType", HK_NULL, HK_NULL, hkClassMember::TYPE_CHAR, hkClassMember::TYPE_VOID, 4, 0, 0, HK_NULL },
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "originalFilename", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxTextureInplaceClass(
		"hkxTextureInplace",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxTextureInplaceClass_Members),
		HK_COUNT_OF(hkxTextureInplaceClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkxIndexBufferIndexTypeEnumItems[] =
	{
		{0, "INDEX_TYPE_INVALID"},
		{1, "INDEX_TYPE_TRI_LIST"},
		{2, "INDEX_TYPE_TRI_STRIP"},
		{3, "INDEX_TYPE_TRI_FAN"},
		{4, "INDEX_TYPE_MAX_ID"},
	};
	static const hkInternalClassEnum hkxIndexBufferEnums[] = {
		{"IndexType", hkxIndexBufferIndexTypeEnumItems, 5, HK_NULL, 0 }
	};
	const hkClassEnum* hkxIndexBufferIndexTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkxIndexBufferEnums[0]);
	static hkInternalClassMember hkxIndexBufferClass_Members[] =
	{
		{ "indexType", HK_NULL, hkxIndexBufferIndexTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "indices16", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "indices32", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "vertexBaseOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "length", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxIndexBufferClass(
		"hkxIndexBuffer",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkxIndexBufferEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkxIndexBufferClass_Members),
		HK_COUNT_OF(hkxIndexBufferClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxMesh_UserChannelInfoClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "className", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxMeshUserChannelInfoClass(
		"hkxMeshUserChannelInfo",
		&hkxAttributeHolderClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxMesh_UserChannelInfoClass_Members),
		HK_COUNT_OF(hkxMesh_UserChannelInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxMeshClass_Members[] =
	{
		{ "sections", &hkxMeshSectionClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "userChannelInfos", &hkxMeshUserChannelInfoClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkxMeshClass(
		"hkxMesh",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxMeshClass_Members),
		HK_COUNT_OF(hkxMeshClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxMeshSectionClass_Members[] =
	{
		{ "vertexBuffer", &hkxVertexBufferClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "indexBuffers", &hkxIndexBufferClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "material", &hkxMaterialClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "userChannels", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_VARIANT, 0, 0, 0, HK_NULL }
	};
	hkClass hkxMeshSectionClass(
		"hkxMeshSection",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxMeshSectionClass_Members),
		HK_COUNT_OF(hkxMeshSectionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexBufferClass_Members[] =
	{
		{ "vertexData", HK_NULL, HK_NULL, hkClassMember::TYPE_HOMOGENEOUSARRAY, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vertexDesc", &hkxVertexDescriptionClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexBufferClass(
		"hkxVertexBuffer",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexBufferClass_Members),
		HK_COUNT_OF(hkxVertexBufferClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkxVertexDescriptionDataTypeEnumItems[] =
	{
		{0, "HKX_DT_NONE"},
		{1, "HKX_DT_UINT8"},
		{2, "HKX_DT_INT16"},
		{3, "HKX_DT_UINT32"},
		{4, "HKX_DT_FLOAT"},
		{5, "HKX_DT_FLOAT2"},
		{6, "HKX_DT_FLOAT3"},
		{7, "HKX_DT_FLOAT4"},
	};
	static const hkInternalClassEnumItem hkxVertexDescriptionDataUsageEnumItems[] =
	{
		{0, "HKX_DU_NONE"},
		{1, "HKX_DU_POSITION"},
		{2, "HKX_DU_COLOR"},
		{4, "HKX_DU_NORMAL"},
		{8, "HKX_DU_TANGENT"},
		{16, "HKX_DU_BINORMAL"},
		{32, "HKX_DU_TEXCOORD"},
		{64, "HKX_DU_BLENDWEIGHTS"},
		{128, "HKX_DU_BLENDINDICES"},
		{256, "HKX_DU_USERDATA"},
	};
	static const hkInternalClassEnum hkxVertexDescriptionEnums[] = {
		{"DataType", hkxVertexDescriptionDataTypeEnumItems, 8, HK_NULL, 0 },
		{"DataUsage", hkxVertexDescriptionDataUsageEnumItems, 10, HK_NULL, 0 }
	};
	const hkClassEnum* hkxVertexDescriptionDataTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkxVertexDescriptionEnums[0]);
	const hkClassEnum* hkxVertexDescriptionDataUsageEnum = reinterpret_cast<const hkClassEnum*>(&hkxVertexDescriptionEnums[1]);
	static hkInternalClassMember hkxVertexDescription_ElementDeclClass_Members[] =
	{
		{ "byteOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, hkxVertexDescriptionDataTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "usage", HK_NULL, hkxVertexDescriptionDataUsageEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexDescriptionElementDeclClass(
		"hkxVertexDescriptionElementDecl",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexDescription_ElementDeclClass_Members),
		HK_COUNT_OF(hkxVertexDescription_ElementDeclClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexDescriptionClass_Members[] =
	{
		{ "stride", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "decls", &hkxVertexDescriptionElementDeclClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexDescriptionClass(
		"hkxVertexDescription",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkxVertexDescriptionEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkxVertexDescriptionClass_Members),
		HK_COUNT_OF(hkxVertexDescriptionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxEdgeSelectionChannelClass_Members[] =
	{
		{ "selectedEdges", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL }
	};
	hkClass hkxEdgeSelectionChannelClass(
		"hkxEdgeSelectionChannel",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxEdgeSelectionChannelClass_Members),
		HK_COUNT_OF(hkxEdgeSelectionChannelClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxTriangleSelectionChannelClass_Members[] =
	{
		{ "selectedTriangles", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL }
	};
	hkClass hkxTriangleSelectionChannelClass(
		"hkxTriangleSelectionChannel",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxTriangleSelectionChannelClass_Members),
		HK_COUNT_OF(hkxTriangleSelectionChannelClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkxVertexFloatDataChannelVertexFloatDimensionsEnumItems[] =
	{
		{0, "FLOAT"},
		{1, "DISTANCE"},
		{2, "ANGLE"},
	};
	static const hkInternalClassEnum hkxVertexFloatDataChannelEnums[] = {
		{"VertexFloatDimensions", hkxVertexFloatDataChannelVertexFloatDimensionsEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkxVertexFloatDataChannelVertexFloatDimensionsEnum = reinterpret_cast<const hkClassEnum*>(&hkxVertexFloatDataChannelEnums[0]);
	static hkInternalClassMember hkxVertexFloatDataChannelClass_Members[] =
	{
		{ "perVertexFloats", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "dimensions", HK_NULL, hkxVertexFloatDataChannelVertexFloatDimensionsEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexFloatDataChannelClass(
		"hkxVertexFloatDataChannel",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkxVertexFloatDataChannelEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkxVertexFloatDataChannelClass_Members),
		HK_COUNT_OF(hkxVertexFloatDataChannelClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexIntDataChannelClass_Members[] =
	{
		{ "perVertexInts", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexIntDataChannelClass(
		"hkxVertexIntDataChannel",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexIntDataChannelClass_Members),
		HK_COUNT_OF(hkxVertexIntDataChannelClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexSelectionChannelClass_Members[] =
	{
		{ "selectedVertices", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexSelectionChannelClass(
		"hkxVertexSelectionChannel",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexSelectionChannelClass_Members),
		HK_COUNT_OF(hkxVertexSelectionChannelClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexVectorDataChannelClass_Members[] =
	{
		{ "perVertexVectors", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexVectorDataChannelClass(
		"hkxVertexVectorDataChannel",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexVectorDataChannelClass_Members),
		HK_COUNT_OF(hkxVertexVectorDataChannelClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4C1T2Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4C1T2Class(
		"hkxVertexP4N4C1T2",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4C1T2Class_Members),
		HK_COUNT_OF(hkxVertexP4N4C1T2Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4Class(
		"hkxVertexP4N4T4",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4C1T6Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4C1T6Class(
		"hkxVertexP4N4C1T6",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4C1T6Class_Members),
		HK_COUNT_OF(hkxVertexP4N4C1T6Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4C1T10Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u4", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v4", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4C1T10Class(
		"hkxVertexP4N4C1T10",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4C1T10Class_Members),
		HK_COUNT_OF(hkxVertexP4N4C1T10Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4B4C1T2Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tangent", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "binormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4B4C1T2Class(
		"hkxVertexP4N4T4B4C1T2",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4B4C1T2Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4B4C1T2Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4B4T4Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tangent", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "binormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4B4T4Class(
		"hkxVertexP4N4T4B4T4",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4B4T4Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4B4T4Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4B4C1T6Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tangent", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "binormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4B4C1T6Class(
		"hkxVertexP4N4T4B4C1T6",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4B4C1T6Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4B4C1T6Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4B4C1T10Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tangent", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "binormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u4", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v4", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4B4C1T10Class(
		"hkxVertexP4N4T4B4C1T10",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4B4C1T10Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4B4C1T10Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4B4W4I4C1Q2Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tangent", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "binormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "weights", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "indices", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "qu", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "qv", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4B4W4I4C1Q2Class(
		"hkxVertexP4N4T4B4W4I4C1Q2",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4B4W4I4C1Q2Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4B4W4I4C1Q2Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4B4W4I4Q4Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tangent", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "binormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "weights", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "indices", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "qu0", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "qu1", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4B4W4I4Q4Class(
		"hkxVertexP4N4T4B4W4I4Q4",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4B4W4I4Q4Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4B4W4I4Q4Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4B4W4I4C1T4Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tangent", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "binormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4B4W4I4C1T4Class(
		"hkxVertexP4N4T4B4W4I4C1T4",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4B4W4I4C1T4Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4B4W4I4C1T4Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4B4W4I4T6Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tangent", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "binormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4B4W4I4T6Class(
		"hkxVertexP4N4T4B4W4I4T6",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4B4W4I4T6Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4B4W4I4T6Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4B4W4I4C1T8Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tangent", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "binormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4B4W4I4C1T8Class(
		"hkxVertexP4N4T4B4W4I4C1T8",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4B4W4I4C1T8Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4B4W4I4C1T8Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4T4B4W4I4C1T12Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tangent", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "binormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u4", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v4", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u5", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v5", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4T4B4W4I4C1T12Class(
		"hkxVertexP4N4T4B4W4I4C1T12",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4T4B4W4I4C1T12Class_Members),
		HK_COUNT_OF(hkxVertexP4N4T4B4W4I4C1T12Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4W4I4C1Q2Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "weights", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "indices", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "qu", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "qv", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4W4I4C1Q2Class(
		"hkxVertexP4N4W4I4C1Q2",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4W4I4C1Q2Class_Members),
		HK_COUNT_OF(hkxVertexP4N4W4I4C1Q2Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4W4I4C1T4Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4W4I4C1T4Class(
		"hkxVertexP4N4W4I4C1T4",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4W4I4C1T4Class_Members),
		HK_COUNT_OF(hkxVertexP4N4W4I4C1T4Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4W4I4T2Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4W4I4T2Class(
		"hkxVertexP4N4W4I4T2",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4W4I4T2Class_Members),
		HK_COUNT_OF(hkxVertexP4N4W4I4T2Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4W4I4T6Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4W4I4T6Class(
		"hkxVertexP4N4W4I4T6",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4W4I4T6Class_Members),
		HK_COUNT_OF(hkxVertexP4N4W4I4T6Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4W4I4C1T8Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4W4I4C1T8Class(
		"hkxVertexP4N4W4I4C1T8",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4W4I4C1T8Class_Members),
		HK_COUNT_OF(hkxVertexP4N4W4I4C1T8Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxVertexP4N4W4I4C1T12Class_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "w3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "i3", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "diffuse", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v0", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v1", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v2", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v3", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u4", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v4", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "u5", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "v5", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxVertexP4N4W4I4C1T12Class(
		"hkxVertexP4N4W4I4C1T12",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxVertexP4N4W4I4C1T12Class_Members),
		HK_COUNT_OF(hkxVertexP4N4W4I4C1T12Class_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxSceneClass_Members[] =
	{
		{ "modeller", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "asset", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sceneLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rootNode", &hkxNodeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "selectionSets", &hkxNodeSelectionSetClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "cameras", &hkxCameraClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "lights", &hkxLightClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "meshes", &hkxMeshClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "materials", &hkxMaterialClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "inplaceTextures", &hkxTextureInplaceClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "externalTextures", &hkxTextureFileClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "skinBindings", &hkxSkinBindingClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "appliedTransform", HK_NULL, HK_NULL, hkClassMember::TYPE_MATRIX3, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkxScene_DefaultStruct
		{
			int s_defaultOffsets[13];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkMatrix3 m_appliedTransform;
		};
		const hkxScene_DefaultStruct hkxScene_Default =
		{
			{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkxScene_DefaultStruct,m_appliedTransform)},
			{1,0,0,0,0,1,0,0,0,0,1,0}
		};
	}
	hkClass hkxSceneClass(
		"hkxScene",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxSceneClass_Members),
		HK_COUNT_OF(hkxSceneClass_Members),
		&hkxScene_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxNodeSelectionSetClass_Members[] =
	{
		{ "selectedNodes", &hkxNodeClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxNodeSelectionSetClass(
		"hkxNodeSelectionSet",
		&hkxAttributeHolderClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxNodeSelectionSetClass_Members),
		HK_COUNT_OF(hkxNodeSelectionSetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkxSkinBindingClass_Members[] =
	{
		{ "mesh", &hkxMeshClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "mapping", &hkxNodeClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "bindPose", HK_NULL, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_MATRIX4, 0, 0, 0, HK_NULL },
		{ "initSkinTransform", HK_NULL, HK_NULL, hkClassMember::TYPE_MATRIX4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkxSkinBindingClass(
		"hkxSkinBinding",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkxSkinBindingClass_Members),
		HK_COUNT_OF(hkxSkinBindingClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkPackfileHeaderClass_Members[] =
	{
		{ "magic", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "userTag", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fileVersion", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "layoutRules", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 4, 0, 0, HK_NULL },
		{ "numSections", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contentsSectionIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contentsSectionOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contentsClassNameSectionIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contentsClassNameSectionOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contentsVersion", HK_NULL, HK_NULL, hkClassMember::TYPE_CHAR, hkClassMember::TYPE_VOID, 16, 0, 0, HK_NULL },
		{ "pad", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL }
	};
	hkClass hkPackfileHeaderClass(
		"hkPackfileHeader",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkPackfileHeaderClass_Members),
		HK_COUNT_OF(hkPackfileHeaderClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkPackfileSectionHeaderClass_Members[] =
	{
		{ "sectionTag", HK_NULL, HK_NULL, hkClassMember::TYPE_CHAR, hkClassMember::TYPE_VOID, 19, 0, 0, HK_NULL },
		{ "nullByte", HK_NULL, HK_NULL, hkClassMember::TYPE_CHAR, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "absoluteDataStart", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "localFixupsOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "globalFixupsOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "virtualFixupsOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "exportsOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "importsOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "endOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkPackfileSectionHeaderClass(
		"hkPackfileSectionHeader",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkPackfileSectionHeaderClass_Members),
		HK_COUNT_OF(hkPackfileSectionHeaderClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkResourceBaseTypeEnumItems[] =
	{
		{0, "TYPE_RESOURCE"},
		{1, "TYPE_CONTAINER"},
	};
	static const hkInternalClassEnum hkResourceBaseEnums[] = {
		{"Type", hkResourceBaseTypeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkResourceBaseTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkResourceBaseEnums[0]);
	hkClass hkResourceBaseClass(
		"hkResourceBase",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkResourceBaseEnums),
		1,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkResourceHandleClass(
		"hkResourceHandle",
		&hkResourceBaseClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkResourceContainerClass(
		"hkResourceContainer",
		&hkResourceBaseClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkMemoryResourceHandle_ExternalLinkClass_Members[] =
	{
		{ "memberName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "externalId", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "externalIdIsAllocated", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "memberNameIsAllocated", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkMemoryResourceHandleExternalLinkClass(
		"hkMemoryResourceHandleExternalLink",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkMemoryResourceHandle_ExternalLinkClass_Members),
		HK_COUNT_OF(hkMemoryResourceHandle_ExternalLinkClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkMemoryResourceHandleClass_Members[] =
	{
		{ "variant", HK_NULL, HK_NULL, hkClassMember::TYPE_VARIANT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "objectIsRerencedObject", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "nameIsAllocated", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "references", &hkMemoryResourceHandleExternalLinkClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkMemoryResourceHandleClass(
		"hkMemoryResourceHandle",
		&hkResourceHandleClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkMemoryResourceHandleClass_Members),
		HK_COUNT_OF(hkMemoryResourceHandleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkMemoryResourceContainerClass_Members[] =
	{
		{ "nameIsAllocated", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "parent", &hkMemoryResourceContainerClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "resourceHandles", &hkMemoryResourceHandleClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "children", &hkMemoryResourceContainerClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkMemoryResourceContainerClass(
		"hkMemoryResourceContainer",
		&hkResourceContainerClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkMemoryResourceContainerClass_Members),
		HK_COUNT_OF(hkMemoryResourceContainerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkRootLevelContainer_NamedVariantClass_Members[] =
	{
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "className", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "variant", HK_NULL, HK_NULL, hkClassMember::TYPE_VARIANT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkRootLevelContainerNamedVariantClass(
		"hkRootLevelContainerNamedVariant",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkRootLevelContainer_NamedVariantClass_Members),
		HK_COUNT_OF(hkRootLevelContainer_NamedVariantClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkRootLevelContainerClass_Members[] =
	{
		{ "namedVariants", &hkRootLevelContainerNamedVariantClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkRootLevelContainerClass(
		"hkRootLevelContainer",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkRootLevelContainerClass_Members),
		HK_COUNT_OF(hkRootLevelContainerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdSplitShapeClass_Members[] =
	{
		{ "graphicsShape", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "geometry", &hkdGeometryClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkdSplitShapeClass(
		"hkdSplitShape",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdSplitShapeClass_Members),
		HK_COUNT_OF(hkdSplitShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdGeometryInvalidUserIdentEnumItems[] =
	{
		{65535/*0xffff*/, "INVALID_USER_IDENT"},
	};
	static const hkInternalClassEnum hkdGeometryEnums[] = {
		{"InvalidUserIdent", hkdGeometryInvalidUserIdentEnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hkdGeometryInvalidUserIdentEnum = reinterpret_cast<const hkClassEnum*>(&hkdGeometryEnums[0]);
	static hkInternalClassMember hkdGeometry_FaceIdentifierClass_Members[] =
	{
		{ "userData", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 4, 0, 0, HK_NULL }
	};
	hkClass hkdGeometryFaceIdentifierClass(
		"hkdGeometryFaceIdentifier",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdGeometry_FaceIdentifierClass_Members),
		HK_COUNT_OF(hkdGeometry_FaceIdentifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdGeometry_FaceClass_Members[] =
	{
		{ "support", &hkdGeometryFaceIdentifierClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "startTriangleIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numTriangles", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "objectIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "parentFaceIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkdGeometryFaceClass(
		"hkdGeometryFace",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdGeometry_FaceClass_Members),
		HK_COUNT_OF(hkdGeometry_FaceClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdGeometryTriangleEdgeMarkerEnumItems[] =
	{
		{65535/*0xffff*/, "UNCONNECTED_EDGE"},
	};
	static const hkInternalClassEnum hkdGeometryTriangleEnums[] = {
		{"EdgeMarker", hkdGeometryTriangleEdgeMarkerEnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hkdGeometryTriangleEdgeMarkerEnum = reinterpret_cast<const hkClassEnum*>(&hkdGeometryTriangleEnums[0]);
	static hkInternalClassMember hkdGeometry_TriangleClass_Members[] =
	{
		{ "vertexIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL },
		{ "neighbouringFaces", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL }
	};
	hkClass hkdGeometryTriangleClass(
		"hkdGeometryTriangle",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdGeometryTriangleEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkdGeometry_TriangleClass_Members),
		HK_COUNT_OF(hkdGeometry_TriangleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdGeometry_ObjectIdentifierClass_Members[] =
	{
		{ "transform", HK_NULL, HK_NULL, hkClassMember::TYPE_MATRIX4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isInverted", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkdGeometryObjectIdentifierClass(
		"hkdGeometryObjectIdentifier",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdGeometry_ObjectIdentifierClass_Members),
		HK_COUNT_OF(hkdGeometry_ObjectIdentifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdGeometryClass_Members[] =
	{
		{ "faces", &hkdGeometryFaceClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "triangles", &hkdGeometryTriangleClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "vertices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL },
		{ "objectIds", &hkdGeometryObjectIdentifierClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "parent", &hkdGeometryClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkdGeometryClass(
		"hkdGeometry",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdGeometryEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkdGeometryClass_Members),
		HK_COUNT_OF(hkdGeometryClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdPropertiesClass_Members[] =
	{
		{ "properties", &hkpPropertyClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkdPropertiesClass(
		"hkdProperties",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdPropertiesClass_Members),
		HK_COUNT_OF(hkdPropertiesClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdBreakableBodyBodyTypeEnumItems[] =
	{
		{0, "BODY_TYPE_INVALID"},
		{1, "BODY_TYPE_SIMPLE"},
		{2, "BODY_TYPE_EMBEDDED"},
		{3, "BODY_TYPE_NUM_TYPES"},
	};
	static const hkInternalClassEnum hkdBreakableBodyEnums[] = {
		{"BodyType", hkdBreakableBodyBodyTypeEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkdBreakableBodyBodyTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdBreakableBodyEnums[0]);
	static hkInternalClassMember hkdBreakableBody_SmallArraySerializeOverrideTypeClass_Members[] =
	{
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "size", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "capacityAndFlags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkdBreakableBodySmallArraySerializeOverrideTypeClass(
		"hkdBreakableBodySmallArraySerializeOverrideType",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdBreakableBody_SmallArraySerializeOverrideTypeClass_Members),
		HK_COUNT_OF(hkdBreakableBody_SmallArraySerializeOverrideTypeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdBreakableBodyClass_Members[] =
	{
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "attachToNearbyObjects", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "destructionWorld", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "physicsBody", &hkpRigidBodyClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "breakableShape", &hkdBreakableShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "controller", &hkdControllerClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "integritySystem", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "ancesterIntegrityUid", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "constraintStrength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "graphicsBody", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "properties", &hkdPropertiesClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "listeners", &hkdBreakableBodySmallArraySerializeOverrideTypeClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "fixedConnectivity", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkdBreakableBodyClass(
		"hkdBreakableBody",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdBreakableBodyEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkdBreakableBodyClass_Members),
		HK_COUNT_OF(hkdBreakableBodyClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdBodyClass_Members[] =
	{
		{ "parentName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "controller", &hkdControllerClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "findInitialContactPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "attachToNearbyObjects", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "constraintStrength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdBody_DefaultStruct
		{
			int s_defaultOffsets[5];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_findInitialContactPoints;
		};
		const hkdBody_DefaultStruct hkdBody_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkdBody_DefaultStruct,m_findInitialContactPoints),-1,-1},
			true
		};
	}
	hkClass hkdBodyClass(
		"hkdBody",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdBodyClass_Members),
		HK_COUNT_OF(hkdBodyClass_Members),
		&hkdBody_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdShapeConnectivityEnumItems[] =
	{
		{0, "CONNECTIVITY_UNSPECIFIED"},
		{1, "CONNECTIVITY_PARTIAL"},
		{2, "CONNECTIVITY_FULL"},
		{3, "CONNECTIVITY_NONE"},
	};
	static const hkInternalClassEnumItem hkdShapeIntegrityTypeEnumItems[] =
	{
		{0, "INTEGRITY_UNSPECIFIED"},
		{1, "INTEGRITY_NONE"},
		{2, "INTEGRITY_ON"},
		{3, "INTEGRITY_CHILDREN"},
		{4, "INTEGRITY_FIXED"},
	};
	static const hkInternalClassEnum hkdShapeEnums[] = {
		{"Connectivity", hkdShapeConnectivityEnumItems, 4, HK_NULL, 0 },
		{"IntegrityType", hkdShapeIntegrityTypeEnumItems, 5, HK_NULL, 0 }
	};
	const hkClassEnum* hkdShapeConnectivityEnum = reinterpret_cast<const hkClassEnum*>(&hkdShapeEnums[0]);
	const hkClassEnum* hkdShapeIntegrityTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdShapeEnums[1]);
	static hkInternalClassMember hkdShapeClass_Members[] =
	{
		{ "parentName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fracture", &hkdFractureClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "mergeCoplanarTriangles", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "connectivity", HK_NULL, hkdShapeConnectivityEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "dynamicFracture", &hkdFractureClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "strength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "relativeSubpieceStrength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "breakingPropogationRate", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "destructionRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "integrityType", HK_NULL, hkdShapeIntegrityTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "tensionLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdShape_DefaultStruct
		{
			int s_defaultOffsets[11];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_mergeCoplanarTriangles;
			hkReal m_relativeSubpieceStrength;
		};
		const hkdShape_DefaultStruct hkdShape_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkdShape_DefaultStruct,m_mergeCoplanarTriangles),-1,-1,-1,HK_OFFSET_OF(hkdShape_DefaultStruct,m_relativeSubpieceStrength),-1,-1,-1,-1},
			true,1.0f
		};
	}
	hkClass hkdShapeClass(
		"hkdShape",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdShapeEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkdShapeClass_Members),
		HK_COUNT_OF(hkdShapeClass_Members),
		&hkdShape_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdBreakableBodyBlueprintClass_Members[] =
	{
		{ "rigidBodyName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "compoundId", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fracture", &hkdFractureClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "dynamicFracture", &hkdFractureClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "controller", &hkdControllerClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "strength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "relativeSubpieceStrength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "breakingPropogationRate", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "attachToNearbyObjects", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "findInitialContactPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "mergeCoplanarTriangles", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "destructionRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdBreakableBodyBlueprint_DefaultStruct
		{
			int s_defaultOffsets[12];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_relativeSubpieceStrength;
			hkReal m_breakingPropogationRate;
			_hkBool m_findInitialContactPoints;
			_hkBool m_mergeCoplanarTriangles;
			hkReal m_destructionRadius;
		};
		const hkdBreakableBodyBlueprint_DefaultStruct hkdBreakableBodyBlueprint_Default =
		{
			{-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkdBreakableBodyBlueprint_DefaultStruct,m_relativeSubpieceStrength),HK_OFFSET_OF(hkdBreakableBodyBlueprint_DefaultStruct,m_breakingPropogationRate),-1,HK_OFFSET_OF(hkdBreakableBodyBlueprint_DefaultStruct,m_findInitialContactPoints),HK_OFFSET_OF(hkdBreakableBodyBlueprint_DefaultStruct,m_mergeCoplanarTriangles),HK_OFFSET_OF(hkdBreakableBodyBlueprint_DefaultStruct,m_destructionRadius)},
			1.0f,0.5f,true,true,0.5f
		};
	}
	hkClass hkdBreakableBodyBlueprintClass(
		"hkdBreakableBodyBlueprint",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdBreakableBodyBlueprintClass_Members),
		HK_COUNT_OF(hkdBreakableBodyBlueprintClass_Members),
		&hkdBreakableBodyBlueprint_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdCompoundBreakableBodyBlueprintClass_Members[] =
	{
		{ "attachToNearbyObjects", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "controller", &hkdControllerClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "groupName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdCompoundBreakableBodyBlueprint_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			const char* m_groupName;
		};
		const hkdCompoundBreakableBodyBlueprint_DefaultStruct hkdCompoundBreakableBodyBlueprint_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkdCompoundBreakableBodyBlueprint_DefaultStruct,m_groupName)},
			""
		};
	}
	hkClass hkdCompoundBreakableBodyBlueprintClass(
		"hkdCompoundBreakableBodyBlueprint",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdCompoundBreakableBodyBlueprintClass_Members),
		HK_COUNT_OF(hkdCompoundBreakableBodyBlueprintClass_Members),
		&hkdCompoundBreakableBodyBlueprint_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdBreakableShapeShapeTypeEnumItems[] =
	{
		{0, "SHAPE_TYPE_INVALID"},
		{1, "SHAPE_TYPE_SIMPLE"},
		{2, "SHAPE_TYPE_COMPOUND"},
		{3, "SHAPE_TYPE_DEFORMABLE"},
		{4, "SHAPE_TYPE_NUM_TYPES"},
	};
	static const hkInternalClassEnumItem hkdBreakableShapeRecalcChildrenEnumItems[] =
	{
		{0, "RECALC_CHILDREN_BREAKING_THRESHOLDS"},
		{1, "DO_NOT_RECALC_CHILDREN_BREAKING_THRESHOLDS"},
	};
	static const hkInternalClassEnumItem hkdBreakableShapeFlagsEnumItems[] =
	{
		{1/*1<<0*/, "FLAG_FIND_INITIAL_CONTACT_POINTS"},
	};
	static const hkInternalClassEnum hkdBreakableShapeEnums[] = {
		{"ShapeType", hkdBreakableShapeShapeTypeEnumItems, 5, HK_NULL, 0 },
		{"RecalcChildren", hkdBreakableShapeRecalcChildrenEnumItems, 2, HK_NULL, 0 },
		{"Flags", hkdBreakableShapeFlagsEnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hkdBreakableShapeShapeTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdBreakableShapeEnums[0]);
	const hkClassEnum* hkdBreakableShapeRecalcChildrenEnum = reinterpret_cast<const hkClassEnum*>(&hkdBreakableShapeEnums[1]);
	const hkClassEnum* hkdBreakableShapeFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkdBreakableShapeEnums[2]);
	static hkInternalClassMember hkdBreakableShape_ConnectionClass_Members[] =
	{
		{ "pivotA", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pivotB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "separatingNormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contactArea", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "a", &hkdBreakableShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "b", &hkdBreakableShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkdBreakableShapeConnectionClass(
		"hkdBreakableShapeConnection",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdBreakableShape_ConnectionClass_Members),
		HK_COUNT_OF(hkdBreakableShape_ConnectionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdBreakableShapeClass_Members[] =
	{
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "connectivityType", HK_NULL, hkdShapeConnectivityEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "integrityType", HK_NULL, hkdShapeIntegrityTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "flags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "parent", &hkdBreakableShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "children", &hkdShapeInstanceInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "connections", &hkdBreakableShapeConnectionClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "massProps", &hkpMassPropertiesClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "physicsShape", &hkpShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "geometry", &hkdGeometryClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "graphicsShape", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "graphicsShapeName", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "properties", &hkdPropertiesClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "dynamicFracture", &hkdFractureClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "strength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "relativeSubpieceStrength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tensionLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "referenceShapeVolume", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minDestructionRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "breakingPropogationRate", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "unusedPadding", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 2, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkdBreakableShape_DefaultStruct
		{
			int s_defaultOffsets[21];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint8 m_flags;
		};
		const hkdBreakableShape_DefaultStruct hkdBreakableShape_Default =
		{
			{-1,-1,-1,HK_OFFSET_OF(hkdBreakableShape_DefaultStruct,m_flags),-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
			0x01
		};
	}
	hkClass hkdBreakableShapeClass(
		"hkdBreakableShape",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdBreakableShapeEnums),
		3,
		reinterpret_cast<const hkClassMember*>(hkdBreakableShapeClass_Members),
		HK_COUNT_OF(hkdBreakableShapeClass_Members),
		&hkdBreakableShape_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdShapeInstanceInfo_RuntimeInfoClass_Members[] =
	{
		{ "distanceToDestructionPoint", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "oldChildIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "forceFixed", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkdShapeInstanceInfoRuntimeInfoClass(
		"hkdShapeInstanceInfoRuntimeInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdShapeInstanceInfo_RuntimeInfoClass_Members),
		HK_COUNT_OF(hkdShapeInstanceInfo_RuntimeInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdShapeInstanceInfoClass_Members[] =
	{
		{ "transform", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "shape", &hkdBreakableShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "damage", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "runtimeInfo", &hkdShapeInstanceInfoRuntimeInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkdShapeInstanceInfoClass(
		"hkdShapeInstanceInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdShapeInstanceInfoClass_Members),
		HK_COUNT_OF(hkdShapeInstanceInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdCompoundBreakableShapeConstructorFlagsEnumItems[] =
	{
		{0, "CTR_FLAGS_NORMAL"},
		{1, "CTR_FLAGS_ADD_EXTRA_PCONVEX_TRANSFORM_SHAPE"},
		{2, "CTR_FLAGS_SET_CHILD_PARENT"},
	};
	static const hkInternalClassEnum hkdCompoundBreakableShapeEnums[] = {
		{"ConstructorFlags", hkdCompoundBreakableShapeConstructorFlagsEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkdCompoundBreakableShapeConstructorFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkdCompoundBreakableShapeEnums[0]);
	static hkInternalClassMember hkdCompoundBreakableShapeClass_Members[] =
	{
		{ "rootBreakableShape", &hkdBreakableShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "useChildrenBreakableThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "unusedPaddingCompound", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 2, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkdCompoundBreakableShapeClass(
		"hkdCompoundBreakableShape",
		&hkdBreakableShapeClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdCompoundBreakableShapeEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkdCompoundBreakableShapeClass_Members),
		HK_COUNT_OF(hkdCompoundBreakableShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdDeformableBreakableShapeClass_Members[] =
	{
		{ "origChildTransforms", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_TRANSFORM, 0, 0, 0, HK_NULL }
	};
	hkClass hkdDeformableBreakableShapeClass(
		"hkdDeformableBreakableShape",
		&hkdCompoundBreakableShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdDeformableBreakableShapeClass_Members),
		HK_COUNT_OF(hkdDeformableBreakableShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdControllerControllerTypeEnumItems[] =
	{
		{0, "CONTROLLER_TYPE_INVALID"},
		{1, "CONTROLLER_TYPE_CONTACTREGION"},
		{2, "CONTROLLER_TYPE_DEFORMATION"},
		{3, "CONTROLLER_TYPE_SPLITINHALF"},
		{4, "CONTROLLER_TYPE_WOOD"},
		{5, "CONTROLLER_TYPE_USER"},
		{6, "CONTROLLER_TYPE_NUM_TYPES"},
	};
	static const hkInternalClassEnum hkdControllerEnums[] = {
		{"ControllerType", hkdControllerControllerTypeEnumItems, 7, HK_NULL, 0 }
	};
	const hkClassEnum* hkdControllerControllerTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdControllerEnums[0]);
	static hkInternalClassMember hkdControllerClass_Members[] =
	{
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "unusedPadding", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkdControllerClass(
		"hkdController",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdControllerEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkdControllerClass_Members),
		HK_COUNT_OF(hkdControllerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdContactRegionControllerClass_Members[] =
	{
		{ "maxRecursionLevels", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dynamicFractureBridge", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkdContactRegionController_DefaultStruct
		{
			int s_defaultOffsets[2];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_maxRecursionLevels;
		};
		const hkdContactRegionController_DefaultStruct hkdContactRegionController_Default =
		{
			{HK_OFFSET_OF(hkdContactRegionController_DefaultStruct,m_maxRecursionLevels),-1},
			1000
		};
	}
	hkClass hkdContactRegionControllerClass(
		"hkdContactRegionController",
		&hkdControllerClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdContactRegionControllerClass_Members),
		HK_COUNT_OF(hkdContactRegionControllerClass_Members),
		&hkdContactRegionController_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdDeformationControllerSkinningTypeEnumItems[] =
	{
		{0, "USER_DEFINED"},
		{1, "DISTANCE_TO_CENTER"},
		{2, "DISTANCE_TO_SURFACE"},
	};
	static const hkInternalClassEnum hkdDeformationControllerEnums[] = {
		{"SkinningType", hkdDeformationControllerSkinningTypeEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkdDeformationControllerSkinningTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdDeformationControllerEnums[0]);
	static hkInternalClassMember hkdDeformationControllerClass_Members[] =
	{
		{ "numSmoothingSteps", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "smoothingRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "softness", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxDeformationDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deformationTau", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "skiningType", HK_NULL, hkdDeformationControllerSkinningTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "skinningSmoothing", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdDeformationController_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint8 m_numSmoothingSteps;
			hkUint8 m_smoothingRadius;
			hkReal m_softness;
			hkReal m_maxDeformationDistance;
			hkReal m_deformationTau;
			hkUint8 /* hkEnum<SkinningType, hkUint8> */ m_skiningType;
			hkReal m_skinningSmoothing;
		};
		const hkdDeformationController_DefaultStruct hkdDeformationController_Default =
		{
			{HK_OFFSET_OF(hkdDeformationController_DefaultStruct,m_numSmoothingSteps),HK_OFFSET_OF(hkdDeformationController_DefaultStruct,m_smoothingRadius),HK_OFFSET_OF(hkdDeformationController_DefaultStruct,m_softness),HK_OFFSET_OF(hkdDeformationController_DefaultStruct,m_maxDeformationDistance),HK_OFFSET_OF(hkdDeformationController_DefaultStruct,m_deformationTau),HK_OFFSET_OF(hkdDeformationController_DefaultStruct,m_skiningType),HK_OFFSET_OF(hkdDeformationController_DefaultStruct,m_skinningSmoothing)},
			1,2,0.6f,10.0f,1.0f,2,0.3f
		};
	}
	hkClass hkdDeformationControllerClass(
		"hkdDeformationController",
		&hkdControllerClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdDeformationControllerEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkdDeformationControllerClass_Members),
		HK_COUNT_OF(hkdDeformationControllerClass_Members),
		&hkdDeformationController_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdWoodControllerClass_Members[] =
	{
		{ "deformationFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deformationStrength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxDeformationDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "applyDeformationOnAllObjects", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numSmoothingSteps", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "smoothingRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dynamicFractureBridge", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	namespace
	{
		struct hkdWoodController_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_deformationFriction;
			hkReal m_deformationStrength;
			hkReal m_maxDeformationDistance;
			hkUint8 m_numSmoothingSteps;
			hkUint8 m_smoothingRadius;
		};
		const hkdWoodController_DefaultStruct hkdWoodController_Default =
		{
			{HK_OFFSET_OF(hkdWoodController_DefaultStruct,m_deformationFriction),HK_OFFSET_OF(hkdWoodController_DefaultStruct,m_deformationStrength),HK_OFFSET_OF(hkdWoodController_DefaultStruct,m_maxDeformationDistance),-1,HK_OFFSET_OF(hkdWoodController_DefaultStruct,m_numSmoothingSteps),HK_OFFSET_OF(hkdWoodController_DefaultStruct,m_smoothingRadius),-1},
			0.5f,10.0f,10.0f,1,2
		};
	}
	hkClass hkdWoodControllerClass(
		"hkdWoodController",
		&hkdControllerClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdWoodControllerClass_Members),
		HK_COUNT_OF(hkdWoodControllerClass_Members),
		&hkdWoodController_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdFractureTypeEnumItems[] =
	{
		{0, "TYPE_INVALID"},
		{1, "TYPE_SPLITINHALF"},
		{2, "TYPE_WOOD"},
		{3, "TYPE_RANDOMSPLIT"},
		{4, "TYPE_SLICE"},
		{5, "TYPE_PIE"},
		{6, "TYPE_NUM_TYPES"},
	};
	static const hkInternalClassEnumItem hkdFractureConnectivityTypeEnumItems[] =
	{
		{0, "CONNECTIVITY_NONE"},
		{1, "CONNECTIVITY_PARTIAL"},
		{2, "CONNECTIVITY_FULL"},
	};
	static const hkInternalClassEnumItem hkdFractureRefitPhysicsTypeEnumItems[] =
	{
		{0, "REFIT_NONE"},
		{1, "REFIT_CONVEX_HULL"},
	};
	static const hkInternalClassEnum hkdFractureEnums[] = {
		{"Type", hkdFractureTypeEnumItems, 7, HK_NULL, 0 },
		{"ConnectivityType", hkdFractureConnectivityTypeEnumItems, 3, HK_NULL, 0 },
		{"RefitPhysicsType", hkdFractureRefitPhysicsTypeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkdFractureTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdFractureEnums[0]);
	const hkClassEnum* hkdFractureConnectivityTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdFractureEnums[1]);
	const hkClassEnum* hkdFractureRefitPhysicsTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdFractureEnums[2]);
	static hkInternalClassMember hkdFractureClass_Members[] =
	{
		{ "rootToLeafRatio", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minimumSize", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxDistanceForConnection", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "refitPhysicsShapes", HK_NULL, hkdFractureRefitPhysicsTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "connectivityType", HK_NULL, hkdFractureConnectivityTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdFracture_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_rootToLeafRatio;
			hkReal m_maxDistanceForConnection;
			hkUint8 /* hkEnum<hkdFracture::RefitPhysicsType, hkUint8> */ m_refitPhysicsShapes;
		};
		const hkdFracture_DefaultStruct hkdFracture_Default =
		{
			{HK_OFFSET_OF(hkdFracture_DefaultStruct,m_rootToLeafRatio),-1,HK_OFFSET_OF(hkdFracture_DefaultStruct,m_maxDistanceForConnection),-1,HK_OFFSET_OF(hkdFracture_DefaultStruct,m_refitPhysicsShapes),-1,-1},
			1e6f,0.1f,1
		};
	}
	hkClass hkdFractureClass(
		"hkdFracture",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdFractureEnums),
		3,
		reinterpret_cast<const hkClassMember*>(hkdFractureClass_Members),
		HK_COUNT_OF(hkdFractureClass_Members),
		&hkdFracture_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdPieFractureClass_Members[] =
	{
		{ "splitGeometry", &hkdGeometryClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "splitGeometryScale", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "splitCentralAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "splitCenter", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numParts", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "splittingPlaneConvexRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdPieFracture_DefaultStruct
		{
			int s_defaultOffsets[6];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_splitGeometryScale;
			_hkVector4 m_splitCentralAxis;
			_hkVector4 m_splitCenter;
			int m_numParts;
		};
		const hkdPieFracture_DefaultStruct hkdPieFracture_Default =
		{
			{-1,HK_OFFSET_OF(hkdPieFracture_DefaultStruct,m_splitGeometryScale),HK_OFFSET_OF(hkdPieFracture_DefaultStruct,m_splitCentralAxis),HK_OFFSET_OF(hkdPieFracture_DefaultStruct,m_splitCenter),HK_OFFSET_OF(hkdPieFracture_DefaultStruct,m_numParts),-1},
		{1.0f,1.0f,1.0f},	{0.0f,1.0f,0.0f},	{0.0f,0.0f,0.0f},8
		};
	}
	hkClass hkdPieFractureClass(
		"hkdPieFracture",
		&hkdFractureClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdPieFractureClass_Members),
		HK_COUNT_OF(hkdPieFractureClass_Members),
		&hkdPieFracture_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdRandomSplitFractureClass_Members[] =
	{
		{ "randomSeed", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "splitLargestVolumesFirst", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "splitPlaneGeometry", &hkdGeometryClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "splitGeometryScale", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numObjectsOnLevel1", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numObjectsOnLevel2", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numObjectsOnLevel3", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numObjectsOnLevel4", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdRandomSplitFracture_DefaultStruct
		{
			int s_defaultOffsets[8];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_randomSeed;
			_hkBool m_splitLargestVolumesFirst;
			_hkVector4 m_splitGeometryScale;
			int m_numObjectsOnLevel1;
		};
		const hkdRandomSplitFracture_DefaultStruct hkdRandomSplitFracture_Default =
		{
			{HK_OFFSET_OF(hkdRandomSplitFracture_DefaultStruct,m_randomSeed),HK_OFFSET_OF(hkdRandomSplitFracture_DefaultStruct,m_splitLargestVolumesFirst),-1,HK_OFFSET_OF(hkdRandomSplitFracture_DefaultStruct,m_splitGeometryScale),HK_OFFSET_OF(hkdRandomSplitFracture_DefaultStruct,m_numObjectsOnLevel1),-1,-1,-1},
		123,true,	{1.0f,1.0f,1.0f},2
		};
	}
	hkClass hkdRandomSplitFractureClass(
		"hkdRandomSplitFracture",
		&hkdFractureClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdRandomSplitFractureClass_Members),
		HK_COUNT_OF(hkdRandomSplitFractureClass_Members),
		&hkdRandomSplitFracture_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdSliceFractureSnapTypeEnumItems[] =
	{
		{0, "SNAP_TO_PIVOT"},
		{1, "SNAP_TO_AABB"},
	};
	static const hkInternalClassEnum hkdSliceFractureEnums[] = {
		{"SnapType", hkdSliceFractureSnapTypeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkdSliceFractureSnapTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdSliceFractureEnums[0]);
	static hkInternalClassMember hkdSliceFractureClass_Members[] =
	{
		{ "splitGeometry", &hkdGeometryClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "splitGeometryScale", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "splittingPlaneNormal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numSubparts", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "snap", HK_NULL, hkdSliceFractureSnapTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "splittingPlaneConvexRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "childFracture", &hkdFractureClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdSliceFracture_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_splitGeometryScale;
			_hkVector4 m_splittingPlaneNormal;
			hkReal m_numSubparts;
		};
		const hkdSliceFracture_DefaultStruct hkdSliceFracture_Default =
		{
			{-1,HK_OFFSET_OF(hkdSliceFracture_DefaultStruct,m_splitGeometryScale),HK_OFFSET_OF(hkdSliceFracture_DefaultStruct,m_splittingPlaneNormal),HK_OFFSET_OF(hkdSliceFracture_DefaultStruct,m_numSubparts),-1,-1,-1},
		{1.0f,1.0f,1.0f},	{0.0f,1.0f,0.0f},2.0f
		};
	}
	hkClass hkdSliceFractureClass(
		"hkdSliceFracture",
		&hkdFractureClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdSliceFractureEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkdSliceFractureClass_Members),
		HK_COUNT_OF(hkdSliceFractureClass_Members),
		&hkdSliceFracture_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdSplitInHalfFractureNumChildrenEnumItems[] =
	{
		{0, "CHILDREN_2"},
		{1, "CHILDREN_4"},
		{2, "CHILDREN_8"},
		{3, "CHILDREN_16"},
		{4, "CHILDREN_32"},
		{5, "CHILDREN_64"},
		{6, "CHILDREN_128"},
		{7, "CHILDREN_256"},
	};
	static const hkInternalClassEnum hkdSplitInHalfFractureEnums[] = {
		{"NumChildren", hkdSplitInHalfFractureNumChildrenEnumItems, 8, HK_NULL, 0 }
	};
	const hkClassEnum* hkdSplitInHalfFractureNumChildrenEnum = reinterpret_cast<const hkClassEnum*>(&hkdSplitInHalfFractureEnums[0]);
	static hkInternalClassMember hkdSplitInHalfFractureClass_Members[] =
	{
		{ "splitPlaneGeometry", &hkdGeometryClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "splitGeometryScale", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numChildren", HK_NULL, hkdSplitInHalfFractureNumChildrenEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "flattenHierarchy", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdSplitInHalfFracture_DefaultStruct
		{
			int s_defaultOffsets[4];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_splitGeometryScale;
			hkUint8 /* hkEnum<NumChildren,hkUint8> */ m_numChildren;
			_hkBool m_flattenHierarchy;
		};
		const hkdSplitInHalfFracture_DefaultStruct hkdSplitInHalfFracture_Default =
		{
			{-1,HK_OFFSET_OF(hkdSplitInHalfFracture_DefaultStruct,m_splitGeometryScale),HK_OFFSET_OF(hkdSplitInHalfFracture_DefaultStruct,m_numChildren),HK_OFFSET_OF(hkdSplitInHalfFracture_DefaultStruct,m_flattenHierarchy)},
			{1.0f,1.0f,1.0f},4,true
		};
	}
	hkClass hkdSplitInHalfFractureClass(
		"hkdSplitInHalfFracture",
		&hkdFractureClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdSplitInHalfFractureEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkdSplitInHalfFractureClass_Members),
		HK_COUNT_OF(hkdSplitInHalfFractureClass_Members),
		&hkdSplitInHalfFracture_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdWoodFractureWidthTypeEnumItems[] =
	{
		{0, "WIDTH_IS_RELATIVE"},
		{1, "WIDTH_IS_ABSOLUTE"},
	};
	static const hkInternalClassEnum hkdWoodFractureEnums[] = {
		{"WidthType", hkdWoodFractureWidthTypeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkdWoodFractureWidthTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdWoodFractureEnums[0]);
	static hkInternalClassMember hkdWoodFracture_SplittingDataClass_Members[] =
	{
		{ "splitGeom", &hkdGeometryClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "splittingAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numSubparts", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "widthRange", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "scale", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "scaleRange", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "splitGeomShiftRangeY", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "splitGeomShiftRangeZ", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "surfaceNormalShearingRange", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fractureLineShearingRange", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fractureNormalShearingRange", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdWoodFractureSplittingData_DefaultStruct
		{
			int s_defaultOffsets[11];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_splittingAxis;
			hkReal m_numSubparts;
			_hkVector4 m_scale;
			_hkVector4 m_scaleRange;
		};
		const hkdWoodFractureSplittingData_DefaultStruct hkdWoodFractureSplittingData_Default =
		{
			{-1,HK_OFFSET_OF(hkdWoodFractureSplittingData_DefaultStruct,m_splittingAxis),HK_OFFSET_OF(hkdWoodFractureSplittingData_DefaultStruct,m_numSubparts),-1,HK_OFFSET_OF(hkdWoodFractureSplittingData_DefaultStruct,m_scale),HK_OFFSET_OF(hkdWoodFractureSplittingData_DefaultStruct,m_scaleRange),-1,-1,-1,-1,-1},
		{0,0,0},3.0f,	{1.0f,1.0f,1.0f},	{0.0f,0.0f,0.0f}
		};
	}
	hkClass hkdWoodFractureSplittingDataClass(
		"hkdWoodFractureSplittingData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdWoodFracture_SplittingDataClass_Members),
		HK_COUNT_OF(hkdWoodFracture_SplittingDataClass_Members),
		&hkdWoodFractureSplittingData_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdWoodFractureClass_Members[] =
	{
		{ "flattenHierarchy", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "randomSeed", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boardData", &hkdWoodFractureSplittingDataClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "splinterData", &hkdWoodFractureSplittingDataClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdWoodFracture_DefaultStruct
		{
			int s_defaultOffsets[4];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_randomSeed;
		};
		const hkdWoodFracture_DefaultStruct hkdWoodFracture_Default =
		{
			{-1,HK_OFFSET_OF(hkdWoodFracture_DefaultStruct,m_randomSeed),-1,-1},
			187
		};
	}
	hkClass hkdWoodFractureClass(
		"hkdWoodFracture",
		&hkdFractureClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdWoodFractureEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkdWoodFractureClass_Members),
		HK_COUNT_OF(hkdWoodFractureClass_Members),
		&hkdWoodFracture_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdDestructionDemoConfigClass_Members[] =
	{
		{ "useData", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initialCharacterPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lightSourcePosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lightSourceDirection", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lightSourceColor", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ambientLightColor", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotateLights", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdDestructionDemoConfig_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_useData;
			_hkVector4 m_initialCharacterPosition;
			_hkVector4 m_lightSourcePosition;
			_hkVector4 m_lightSourceDirection;
			_hkVector4 m_lightSourceColor;
			_hkVector4 m_ambientLightColor;
		};
		const hkdDestructionDemoConfig_DefaultStruct hkdDestructionDemoConfig_Default =
		{
			{HK_OFFSET_OF(hkdDestructionDemoConfig_DefaultStruct,m_useData),HK_OFFSET_OF(hkdDestructionDemoConfig_DefaultStruct,m_initialCharacterPosition),HK_OFFSET_OF(hkdDestructionDemoConfig_DefaultStruct,m_lightSourcePosition),HK_OFFSET_OF(hkdDestructionDemoConfig_DefaultStruct,m_lightSourceDirection),HK_OFFSET_OF(hkdDestructionDemoConfig_DefaultStruct,m_lightSourceColor),HK_OFFSET_OF(hkdDestructionDemoConfig_DefaultStruct,m_ambientLightColor),-1},
	true,	{0.0f,1.0f,10.0f},	{0.0f,0.0f,0.0f},	{0.0f,-3.0f,-1.0f},	{1.0f,1.0f,1.0f,1.0f},	{0.1f,0.1f,0.1f}
		};
	}
	hkClass hkdDestructionDemoConfigClass(
		"hkdDestructionDemoConfig",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdDestructionDemoConfigClass_Members),
		HK_COUNT_OF(hkdDestructionDemoConfigClass_Members),
		&hkdDestructionDemoConfig_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkdWeaponBlueprintTypeEnumItems[] =
	{
		{0, "WEAPON_TYPE_INVALID"},
		{1, "WEAPON_TYPE_BALLGUN"},
		{2, "WEAPON_TYPE_GRENADEGUN"},
		{3, "WEAPON_TYPE_GRAVITYGUN"},
		{4, "WEAPON_TYPE_MOUNTEDBALLGUN"},
		{5, "WEAPON_TYPE_CHANGEMASSGUN"},
		{6, "WEAPON_TYPE_NUM_TYPES"},
	};
	static const hkInternalClassEnumItem hkdWeaponBlueprintKeyboardKeyEnumItems[] =
	{
		{112/*0x70*/, "KEY_F1"},
		{113/*0x71*/, "KEY_F2"},
		{114/*0x72*/, "KEY_F3"},
		{115/*0x73*/, "KEY_F4"},
		{116/*0x74*/, "KEY_F5"},
		{117/*0x75*/, "KEY_F6"},
		{118/*0x76*/, "KEY_F7"},
		{119/*0x77*/, "KEY_F8"},
		{120/*0x78*/, "KEY_F9"},
		{121/*0x79*/, "KEY_F10"},
		{122/*0x7A*/, "KEY_F11"},
		{123/*0x7B*/, "KEY_F12"},
	};
	static const hkInternalClassEnum hkdWeaponBlueprintEnums[] = {
		{"Type", hkdWeaponBlueprintTypeEnumItems, 7, HK_NULL, 0 },
		{"KeyboardKey", hkdWeaponBlueprintKeyboardKeyEnumItems, 12, HK_NULL, 0 }
	};
	const hkClassEnum* hkdWeaponBlueprintTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkdWeaponBlueprintEnums[0]);
	const hkClassEnum* hkdWeaponBlueprintKeyboardKeyEnum = reinterpret_cast<const hkClassEnum*>(&hkdWeaponBlueprintEnums[1]);
	static hkInternalClassMember hkdWeaponBlueprintClass_Members[] =
	{
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "keyboardKey", HK_NULL, hkdWeaponBlueprintKeyboardKeyEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdWeaponBlueprint_DefaultStruct
		{
			int s_defaultOffsets[3];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			const char* m_name;
			hkUint8 /* hkEnum<KeyboardKey, hkUint8> */ m_keyboardKey;
		};
		const hkdWeaponBlueprint_DefaultStruct hkdWeaponBlueprint_Default =
		{
			{-1,HK_OFFSET_OF(hkdWeaponBlueprint_DefaultStruct,m_name),HK_OFFSET_OF(hkdWeaponBlueprint_DefaultStruct,m_keyboardKey)},
			"",0x71
		};
	}
	hkClass hkdWeaponBlueprintClass(
		"hkdWeaponBlueprint",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkdWeaponBlueprintEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkdWeaponBlueprintClass_Members),
		HK_COUNT_OF(hkdWeaponBlueprintClass_Members),
		&hkdWeaponBlueprint_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdBallGunBlueprintClass_Members[] =
	{
		{ "bulletRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bulletVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bulletMass", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damageMultiplier", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxBulletsInWorld", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdBallGunBlueprint_DefaultStruct
		{
			int s_defaultOffsets[5];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_bulletRadius;
			hkReal m_bulletVelocity;
			hkReal m_bulletMass;
			hkReal m_damageMultiplier;
			int m_maxBulletsInWorld;
		};
		const hkdBallGunBlueprint_DefaultStruct hkdBallGunBlueprint_Default =
		{
			{HK_OFFSET_OF(hkdBallGunBlueprint_DefaultStruct,m_bulletRadius),HK_OFFSET_OF(hkdBallGunBlueprint_DefaultStruct,m_bulletVelocity),HK_OFFSET_OF(hkdBallGunBlueprint_DefaultStruct,m_bulletMass),HK_OFFSET_OF(hkdBallGunBlueprint_DefaultStruct,m_damageMultiplier),HK_OFFSET_OF(hkdBallGunBlueprint_DefaultStruct,m_maxBulletsInWorld)},
			0.2f,40.0f,50.0f,50.0f,100
		};
	}
	hkClass hkdBallGunBlueprintClass(
		"hkdBallGunBlueprint",
		&hkdWeaponBlueprintClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdBallGunBlueprintClass_Members),
		HK_COUNT_OF(hkdBallGunBlueprintClass_Members),
		&hkdBallGunBlueprint_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdMountedBallGunBlueprintClass_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdMountedBallGunBlueprint_DefaultStruct
		{
			int s_defaultOffsets[1];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_position;
		};
		const hkdMountedBallGunBlueprint_DefaultStruct hkdMountedBallGunBlueprint_Default =
		{
			{HK_OFFSET_OF(hkdMountedBallGunBlueprint_DefaultStruct,m_position)},
			{0.0f,100.0f,0.0f}
		};
	}
	hkClass hkdMountedBallGunBlueprintClass(
		"hkdMountedBallGunBlueprint",
		&hkdBallGunBlueprintClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdMountedBallGunBlueprintClass_Members),
		HK_COUNT_OF(hkdMountedBallGunBlueprintClass_Members),
		&hkdMountedBallGunBlueprint_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdGrenadeGunBlueprintClass_Members[] =
	{
		{ "maxProjectiles", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "reloadTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdGrenadeGunBlueprint_DefaultStruct
		{
			int s_defaultOffsets[2];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_maxProjectiles;
			hkReal m_reloadTime;
		};
		const hkdGrenadeGunBlueprint_DefaultStruct hkdGrenadeGunBlueprint_Default =
		{
			{HK_OFFSET_OF(hkdGrenadeGunBlueprint_DefaultStruct,m_maxProjectiles),HK_OFFSET_OF(hkdGrenadeGunBlueprint_DefaultStruct,m_reloadTime)},
			5,0.3f
		};
	}
	hkClass hkdGrenadeGunBlueprintClass(
		"hkdGrenadeGunBlueprint",
		&hkdWeaponBlueprintClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdGrenadeGunBlueprintClass_Members),
		HK_COUNT_OF(hkdGrenadeGunBlueprintClass_Members),
		&hkdGrenadeGunBlueprint_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdGravityGunBlueprintClass_Members[] =
	{
		{ "maxNumObjectsPicked", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxMassOfObjectPicked", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxDistOfObjectPicked", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "impulseAppliedWhenObjectNotPicked", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "throwVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "capturedObjectPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "capturedObjectsOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdGravityGunBlueprint_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_maxNumObjectsPicked;
			hkReal m_maxMassOfObjectPicked;
			hkReal m_maxDistOfObjectPicked;
			hkReal m_impulseAppliedWhenObjectNotPicked;
			hkReal m_throwVelocity;
			_hkVector4 m_capturedObjectPosition;
			_hkVector4 m_capturedObjectsOffset;
		};
		const hkdGravityGunBlueprint_DefaultStruct hkdGravityGunBlueprint_Default =
		{
			{HK_OFFSET_OF(hkdGravityGunBlueprint_DefaultStruct,m_maxNumObjectsPicked),HK_OFFSET_OF(hkdGravityGunBlueprint_DefaultStruct,m_maxMassOfObjectPicked),HK_OFFSET_OF(hkdGravityGunBlueprint_DefaultStruct,m_maxDistOfObjectPicked),HK_OFFSET_OF(hkdGravityGunBlueprint_DefaultStruct,m_impulseAppliedWhenObjectNotPicked),HK_OFFSET_OF(hkdGravityGunBlueprint_DefaultStruct,m_throwVelocity),HK_OFFSET_OF(hkdGravityGunBlueprint_DefaultStruct,m_capturedObjectPosition),HK_OFFSET_OF(hkdGravityGunBlueprint_DefaultStruct,m_capturedObjectsOffset)},
	10,200.0f,50.0f,100.0f,40.0f,	{2.5f,0.6f,0.0f},	{0.0f,1.0f,0.0f}
		};
	}
	hkClass hkdGravityGunBlueprintClass(
		"hkdGravityGunBlueprint",
		&hkdWeaponBlueprintClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdGravityGunBlueprintClass_Members),
		HK_COUNT_OF(hkdGravityGunBlueprintClass_Members),
		&hkdGravityGunBlueprint_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkdChangeMassGunBlueprintClass_Members[] =
	{
		{ "massChangeRatio", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxDistOfObjectPicked", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkdChangeMassGunBlueprint_DefaultStruct
		{
			int s_defaultOffsets[2];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_massChangeRatio;
			hkReal m_maxDistOfObjectPicked;
		};
		const hkdChangeMassGunBlueprint_DefaultStruct hkdChangeMassGunBlueprint_Default =
		{
			{HK_OFFSET_OF(hkdChangeMassGunBlueprint_DefaultStruct,m_massChangeRatio),HK_OFFSET_OF(hkdChangeMassGunBlueprint_DefaultStruct,m_maxDistOfObjectPicked)},
			2.0f,50.0f
		};
	}
	hkClass hkdChangeMassGunBlueprintClass(
		"hkdChangeMassGunBlueprint",
		&hkdWeaponBlueprintClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkdChangeMassGunBlueprintClass_Members),
		HK_COUNT_OF(hkdChangeMassGunBlueprintClass_Members),
		&hkdChangeMassGunBlueprint_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpCdBodyClass_Members[] =
	{
		{ "shape", &hkpShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "shapeKey", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "motion", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "parent", &hkpCdBodyClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpCdBodyClass(
		"hkpCdBody",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpCdBodyClass_Members),
		HK_COUNT_OF(hkpCdBodyClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpCollidableForceCollideOntoPpuReasonsEnumItems[] =
	{
		{1, "FORCE_PPU_USER_REQUEST"},
		{2, "FORCE_PPU_SHAPE_REQUEST"},
		{4, "FORCE_PPU_MODIFIER_REQUEST"},
	};
	static const hkInternalClassEnum hkpCollidableEnums[] = {
		{"ForceCollideOntoPpuReasons", hkpCollidableForceCollideOntoPpuReasonsEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkpCollidableForceCollideOntoPpuReasonsEnum = reinterpret_cast<const hkClassEnum*>(&hkpCollidableEnums[0]);
	static hkInternalClassMember hkpCollidable_BoundingVolumeDataClass_Members[] =
	{
		{ "min", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL },
		{ "expansionMin", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL },
		{ "expansionShift", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "max", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL },
		{ "expansionMax", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numChildShapeAabbs", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "capacityChildShapeAabbs", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "childShapeAabbs", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "childShapeKeys", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpCollidableBoundingVolumeDataClass(
		"hkpCollidableBoundingVolumeData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpCollidable_BoundingVolumeDataClass_Members),
		HK_COUNT_OF(hkpCollidable_BoundingVolumeDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpCollidableClass_Members[] =
	{
		{ "ownerOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "forceCollideOntoPpu", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "shapeSizeOnSpu", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "broadPhaseHandle", &hkpTypedBroadPhaseHandleClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "boundingVolumeData", &hkpCollidableBoundingVolumeDataClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "allowedPenetrationDepth", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpCollidableClass(
		"hkpCollidable",
		&hkpCdBodyClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpCollidableEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpCollidableClass_Members),
		HK_COUNT_OF(hkpCollidableClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpTypedBroadPhaseHandleClass_Members[] =
	{
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ownerOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "objectQualityType", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionFilterInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpTypedBroadPhaseHandleClass(
		"hkpTypedBroadPhaseHandle",
		&hkpBroadPhaseHandleClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpTypedBroadPhaseHandleClass_Members),
		HK_COUNT_OF(hkpTypedBroadPhaseHandleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpCollidableCollidableFilterClass(
		"hkpCollidableCollidableFilter",
		HK_NULL,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpCollisionFilterhkpFilterTypeEnumItems[] =
	{
		{0, "HK_FILTER_UNKNOWN"},
		{1, "HK_FILTER_NULL"},
		{2, "HK_FILTER_GROUP"},
		{3, "HK_FILTER_LIST"},
		{4, "HK_FILTER_CUSTOM"},
		{5, "HK_FILTER_PAIR"},
		{6, "HK_FILTER_CONSTRAINT"},
	};
	static const hkInternalClassEnum hkpCollisionFilterEnums[] = {
		{"hkpFilterType", hkpCollisionFilterhkpFilterTypeEnumItems, 7, HK_NULL, 0 }
	};
	const hkClassEnum* hkpCollisionFilterhkpFilterTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpCollisionFilterEnums[0]);
	static hkInternalClassMember hkpCollisionFilterClass_Members[] =
	{
		{ "prepad", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "type", HK_NULL, hkpCollisionFilterhkpFilterTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "postpad", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL }
	};
	hkClass hkpCollisionFilterClass(
		"hkpCollisionFilter",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		4,
		reinterpret_cast<const hkClassEnum*>(hkpCollisionFilterEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpCollisionFilterClass_Members),
		HK_COUNT_OF(hkpCollisionFilterClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpConvexListFilterConvexListCollisionTypeEnumItems[] =
	{
		{0, "TREAT_CONVEX_LIST_AS_NORMAL"},
		{1, "TREAT_CONVEX_LIST_AS_LIST"},
		{2, "TREAT_CONVEX_LIST_AS_CONVEX"},
	};
	static const hkInternalClassEnum hkpConvexListFilterEnums[] = {
		{"ConvexListCollisionType", hkpConvexListFilterConvexListCollisionTypeEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkpConvexListFilterConvexListCollisionTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpConvexListFilterEnums[0]);
	hkClass hkpConvexListFilterClass(
		"hkpConvexListFilter",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpConvexListFilterEnums),
		1,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpRayCollidableFilterClass(
		"hkpRayCollidableFilter",
		HK_NULL,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpShapeCollectionFilterClass(
		"hkpShapeCollectionFilter",
		HK_NULL,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpDefaultConvexListFilterClass(
		"hkpDefaultConvexListFilter",
		&hkpConvexListFilterClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpGroupFilterClass_Members[] =
	{
		{ "nextFreeSystemGroup", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionLookupTable", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 32, 0, 0, HK_NULL },
		{ "pad256", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 4, 0, 0, HK_NULL }
	};
	hkClass hkpGroupFilterClass(
		"hkpGroupFilter",
		&hkpCollisionFilterClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpGroupFilterClass_Members),
		HK_COUNT_OF(hkpGroupFilterClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpCollisionFilterListClass_Members[] =
	{
		{ "collisionFilters", &hkpCollisionFilterClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpCollisionFilterListClass(
		"hkpCollisionFilterList",
		&hkpCollisionFilterClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpCollisionFilterListClass_Members),
		HK_COUNT_OF(hkpCollisionFilterListClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpNullCollisionFilterClass(
		"hkpNullCollisionFilter",
		&hkpCollisionFilterClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpShapeClass_Members[] =
	{
		{ "userData", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "type", HK_NULL, HK_NULL, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpShapeClass(
		"hkpShape",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpShapeClass_Members),
		HK_COUNT_OF(hkpShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpShapeContainerClass(
		"hkpShapeContainer",
		HK_NULL,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSingleShapeContainerClass_Members[] =
	{
		{ "childShape", &hkpShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSingleShapeContainerClass(
		"hkpSingleShapeContainer",
		&hkpShapeContainerClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSingleShapeContainerClass_Members),
		HK_COUNT_OF(hkpSingleShapeContainerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpShapeCollectionCollectionTypeEnumItems[] =
	{
		{0, "COLLECTION_LIST"},
		{1, "COLLECTION_EXTENDED_MESH"},
		{2, "COLLECTION_TRISAMPLED_HEIGHTFIELD"},
		{3, "COLLECTION_USER"},
		{4, "COLLECTION_MAX"},
	};
	static const hkInternalClassEnum hkpShapeCollectionEnums[] = {
		{"CollectionType", hkpShapeCollectionCollectionTypeEnumItems, 5, HK_NULL, 0 }
	};
	const hkClassEnum* hkpShapeCollectionCollectionTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpShapeCollectionEnums[0]);
	static hkInternalClassMember hkpShapeCollectionClass_Members[] =
	{
		{ "disableWelding", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collectionType", HK_NULL, hkpShapeCollectionCollectionTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpShapeCollection_DefaultStruct
		{
			int s_defaultOffsets[2];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint8 /* hkEnum<CollectionType, hkUint8> */ m_collectionType;
		};
		const hkpShapeCollection_DefaultStruct hkpShapeCollection_Default =
		{
			{-1,HK_OFFSET_OF(hkpShapeCollection_DefaultStruct,m_collectionType)},
			3
		};
	}
	hkClass hkpShapeCollectionClass(
		"hkpShapeCollection",
		&hkpShapeClass,
		0,
		HK_NULL,
		1,
		reinterpret_cast<const hkClassEnum*>(hkpShapeCollectionEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpShapeCollectionClass_Members),
		HK_COUNT_OF(hkpShapeCollectionClass_Members),
		&hkpShapeCollection_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpExtendedMeshShapeIndexStridingTypeEnumItems[] =
	{
		{0, "INDICES_INVALID"},
		{1, "INDICES_INT16"},
		{2, "INDICES_INT32"},
		{3, "INDICES_MAX_ID"},
	};
	static const hkInternalClassEnumItem hkpExtendedMeshShapeMaterialIndexStridingTypeEnumItems[] =
	{
		{0, "MATERIAL_INDICES_INVALID"},
		{1, "MATERIAL_INDICES_INT8"},
		{2, "MATERIAL_INDICES_INT16"},
		{3, "MATERIAL_INDICES_MAX_ID"},
	};
	static const hkInternalClassEnumItem hkpExtendedMeshShapeSubpartTypeEnumItems[] =
	{
		{0, "SUBPART_TRIANGLES"},
		{1, "SUBPART_SHAPE"},
	};
	static const hkInternalClassEnum hkpExtendedMeshShapeEnums[] = {
		{"IndexStridingType", hkpExtendedMeshShapeIndexStridingTypeEnumItems, 4, HK_NULL, 0 },
		{"MaterialIndexStridingType", hkpExtendedMeshShapeMaterialIndexStridingTypeEnumItems, 4, HK_NULL, 0 },
		{"SubpartType", hkpExtendedMeshShapeSubpartTypeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkpExtendedMeshShapeIndexStridingTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpExtendedMeshShapeEnums[0]);
	const hkClassEnum* hkpExtendedMeshShapeMaterialIndexStridingTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpExtendedMeshShapeEnums[1]);
	const hkClassEnum* hkpExtendedMeshShapeSubpartTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpExtendedMeshShapeEnums[2]);
	static hkInternalClassMember hkpExtendedMeshShape_SubpartClass_Members[] =
	{
		{ "type", HK_NULL, hkpExtendedMeshShapeSubpartTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "materialIndexStridingType", HK_NULL, hkpExtendedMeshShapeIndexStridingTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "materialStriding", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "materialIndexBase", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "materialIndexStriding", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numMaterials", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "materialBase", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpExtendedMeshShapeSubpartClass(
		"hkpExtendedMeshShapeSubpart",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpExtendedMeshShape_SubpartClass_Members),
		HK_COUNT_OF(hkpExtendedMeshShape_SubpartClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpExtendedMeshShape_TrianglesSubpartClass_Members[] =
	{
		{ "numTriangleShapes", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vertexBase", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "vertexStriding", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numVertices", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extrusion", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "indexBase", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "indexStriding", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "stridingType", HK_NULL, hkpExtendedMeshShapeIndexStridingTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "flipAlternateTriangles", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "triangleOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpExtendedMeshShapeTrianglesSubpart_DefaultStruct
		{
			int s_defaultOffsets[10];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_triangleOffset;
		};
		const hkpExtendedMeshShapeTrianglesSubpart_DefaultStruct hkpExtendedMeshShapeTrianglesSubpart_Default =
		{
			{-1,-1,-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkpExtendedMeshShapeTrianglesSubpart_DefaultStruct,m_triangleOffset)},
			-1
		};
	}
	hkClass hkpExtendedMeshShapeTrianglesSubpartClass(
		"hkpExtendedMeshShapeTrianglesSubpart",
		&hkpExtendedMeshShapeSubpartClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpExtendedMeshShape_TrianglesSubpartClass_Members),
		HK_COUNT_OF(hkpExtendedMeshShape_TrianglesSubpartClass_Members),
		&hkpExtendedMeshShapeTrianglesSubpart_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpExtendedMeshShape_ShapesSubpartClass_Members[] =
	{
		{ "childShapes", &hkpConvexShapeClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "offsetSet", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotationSet", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "childShapesAllocatedInternally", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "rotation", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "translation", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpExtendedMeshShapeShapesSubpartClass(
		"hkpExtendedMeshShapeShapesSubpart",
		&hkpExtendedMeshShapeSubpartClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpExtendedMeshShape_ShapesSubpartClass_Members),
		HK_COUNT_OF(hkpExtendedMeshShape_ShapesSubpartClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpExtendedMeshShapeClass_Members[] =
	{
		{ "embeddedTrianglesSubpart", &hkpExtendedMeshShapeTrianglesSubpartClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "scaling", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aabbHalfExtents", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aabbCenter", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numBitsForSubpartIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "trianglesSubparts", &hkpExtendedMeshShapeTrianglesSubpartClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "trianglesSubpartsAllocatedInternally", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "shapesSubparts", &hkpExtendedMeshShapeShapesSubpartClass, HK_NULL, hkClassMember::TYPE_SIMPLEARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "shapesSubpartsAllocatedInternally", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "weldingInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "weldingType", HK_NULL, hkpWeldingUtilityWeldingTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "triangleRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpExtendedMeshShape_DefaultStruct
		{
			int s_defaultOffsets[12];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint8 /* hkEnum<hkpWeldingUtility::WeldingType, hkUint8> */ m_weldingType;
		};
		const hkpExtendedMeshShape_DefaultStruct hkpExtendedMeshShape_Default =
		{
			{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkpExtendedMeshShape_DefaultStruct,m_weldingType),-1},
			6
		};
	}
	hkClass hkpExtendedMeshShapeClass(
		"hkpExtendedMeshShape",
		&hkpShapeCollectionClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpExtendedMeshShapeEnums),
		3,
		reinterpret_cast<const hkClassMember*>(hkpExtendedMeshShapeClass_Members),
		HK_COUNT_OF(hkpExtendedMeshShapeClass_Members),
		&hkpExtendedMeshShape_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpListShapeListShapeFlagsEnumItems[] =
	{
		{0, "ALL_FLAGS_CLEAR"},
		{1/*1<<0*/, "DISABLE_SPU_CACHE_FOR_LIST_CHILD_INFO"},
	};
	static const hkInternalClassEnum hkpListShapeEnums[] = {
		{"ListShapeFlags", hkpListShapeListShapeFlagsEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkpListShapeListShapeFlagsEnum = reinterpret_cast<const hkClassEnum*>(&hkpListShapeEnums[0]);
	static hkInternalClassMember hkpListShape_ChildInfoClass_Members[] =
	{
		{ "shape", &hkpShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL },
		{ "collisionFilterInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "shapeSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "numChildShapes", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpListShapeChildInfoClass(
		"hkpListShapeChildInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpListShape_ChildInfoClass_Members),
		HK_COUNT_OF(hkpListShape_ChildInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpListShapeClass_Members[] =
	{
		{ "childInfo", &hkpListShapeChildInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "flags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numDisabledChildren", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aabbHalfExtents", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aabbCenter", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "enabledChildren", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 8, 0, 0, HK_NULL }
	};
	hkClass hkpListShapeClass(
		"hkpListShape",
		&hkpShapeCollectionClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpListShapeEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpListShapeClass_Members),
		HK_COUNT_OF(hkpListShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMeshMaterialClass_Members[] =
	{
		{ "filterInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMeshMaterialClass(
		"hkpMeshMaterial",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMeshMaterialClass_Members),
		HK_COUNT_OF(hkpMeshMaterialClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSimpleMeshShape_TriangleClass_Members[] =
	{
		{ "a", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "b", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "c", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "weldingInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSimpleMeshShapeTriangleClass(
		"hkpSimpleMeshShapeTriangle",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSimpleMeshShape_TriangleClass_Members),
		HK_COUNT_OF(hkpSimpleMeshShape_TriangleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSimpleMeshShapeClass_Members[] =
	{
		{ "vertices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL },
		{ "triangles", &hkpSimpleMeshShapeTriangleClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "materialIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "weldingType", HK_NULL, hkpWeldingUtilityWeldingTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpSimpleMeshShape_DefaultStruct
		{
			int s_defaultOffsets[5];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint8 /* hkEnum<hkpWeldingUtility::WeldingType, hkUint8> */ m_weldingType;
		};
		const hkpSimpleMeshShape_DefaultStruct hkpSimpleMeshShape_Default =
		{
			{-1,-1,-1,-1,HK_OFFSET_OF(hkpSimpleMeshShape_DefaultStruct,m_weldingType)},
			6
		};
	}
	hkClass hkpSimpleMeshShapeClass(
		"hkpSimpleMeshShape",
		&hkpShapeCollectionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSimpleMeshShapeClass_Members),
		HK_COUNT_OF(hkpSimpleMeshShapeClass_Members),
		&hkpSimpleMeshShape_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStorageExtendedMeshShape_MeshSubpartStorageClass_Members[] =
	{
		{ "vertices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL },
		{ "indices16", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "indices32", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "materialIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "materials", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "materialIndices16", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL }
	};
	hkClass hkpStorageExtendedMeshShapeMeshSubpartStorageClass(
		"hkpStorageExtendedMeshShapeMeshSubpartStorage",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStorageExtendedMeshShape_MeshSubpartStorageClass_Members),
		HK_COUNT_OF(hkpStorageExtendedMeshShape_MeshSubpartStorageClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStorageExtendedMeshShape_ShapeSubpartStorageClass_Members[] =
	{
		{ "shapes", &hkpConvexShapeClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "materialIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "materials", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "materialIndices16", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL }
	};
	hkClass hkpStorageExtendedMeshShapeShapeSubpartStorageClass(
		"hkpStorageExtendedMeshShapeShapeSubpartStorage",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStorageExtendedMeshShape_ShapeSubpartStorageClass_Members),
		HK_COUNT_OF(hkpStorageExtendedMeshShape_ShapeSubpartStorageClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStorageExtendedMeshShapeClass_Members[] =
	{
		{ "meshstorage", &hkpStorageExtendedMeshShapeMeshSubpartStorageClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "shapestorage", &hkpStorageExtendedMeshShapeShapeSubpartStorageClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpStorageExtendedMeshShapeClass(
		"hkpStorageExtendedMeshShape",
		&hkpExtendedMeshShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStorageExtendedMeshShapeClass_Members),
		HK_COUNT_OF(hkpStorageExtendedMeshShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpBvTreeShapeBvTreeTypeEnumItems[] =
	{
		{0, "BVTREE_MOPP"},
		{1, "BVTREE_TRISAMPLED_HEIGHTFIELD"},
		{2, "BVTREE_USER"},
		{3, "BVTREE_MAX"},
	};
	static const hkInternalClassEnum hkpBvTreeShapeEnums[] = {
		{"BvTreeType", hkpBvTreeShapeBvTreeTypeEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkpBvTreeShapeBvTreeTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpBvTreeShapeEnums[0]);
	static hkInternalClassMember hkpBvTreeShapeClass_Members[] =
	{
		{ "bvTreeType", HK_NULL, hkpBvTreeShapeBvTreeTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpBvTreeShape_DefaultStruct
		{
			int s_defaultOffsets[1];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint8 /* hkEnum<BvTreeType, hkUint8> */ m_bvTreeType;
		};
		const hkpBvTreeShape_DefaultStruct hkpBvTreeShape_Default =
		{
			{HK_OFFSET_OF(hkpBvTreeShape_DefaultStruct,m_bvTreeType)},
			2
		};
	}
	hkClass hkpBvTreeShapeClass(
		"hkpBvTreeShape",
		&hkpShapeClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpBvTreeShapeEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpBvTreeShapeClass_Members),
		HK_COUNT_OF(hkpBvTreeShapeClass_Members),
		&hkpBvTreeShape_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkMoppBvTreeShapeBaseClass_Members[] =
	{
		{ "code", &hkpMoppCodeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "moppData", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "moppDataSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "codeInfoCopy", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkMoppBvTreeShapeBaseClass(
		"hkMoppBvTreeShapeBase",
		&hkpBvTreeShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkMoppBvTreeShapeBaseClass_Members),
		HK_COUNT_OF(hkMoppBvTreeShapeBaseClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMoppBvTreeShapeClass_Members[] =
	{
		{ "child", &hkpSingleShapeContainerClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "childSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpMoppBvTreeShapeClass(
		"hkpMoppBvTreeShape",
		&hkMoppBvTreeShapeBaseClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMoppBvTreeShapeClass_Members),
		HK_COUNT_OF(hkpMoppBvTreeShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpRemoveTerminalsMoppModifierClass_Members[] =
	{
		{ "removeInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "tempShapesToRemove", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpRemoveTerminalsMoppModifierClass(
		"hkpRemoveTerminalsMoppModifier",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpRemoveTerminalsMoppModifierClass_Members),
		HK_COUNT_OF(hkpRemoveTerminalsMoppModifierClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpConvexShapeWeldResultEnumItems[] =
	{
		{0, "WELD_RESULT_REJECT_CONTACT_POINT"},
		{1, "WELD_RESULT_ACCEPT_CONTACT_POINT_MODIFIED"},
		{2, "WELD_RESULT_ACCEPT_CONTACT_POINT_UNMODIFIED"},
	};
	static const hkInternalClassEnum hkpConvexShapeEnums[] = {
		{"WeldResult", hkpConvexShapeWeldResultEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkpConvexShapeWeldResultEnum = reinterpret_cast<const hkClassEnum*>(&hkpConvexShapeEnums[0]);
	static hkInternalClassMember hkpConvexShapeClass_Members[] =
	{
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConvexShapeClass(
		"hkpConvexShape",
		&hkpSphereRepShapeClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpConvexShapeEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpConvexShapeClass_Members),
		HK_COUNT_OF(hkpConvexShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConvexTransformShapeBaseClass_Members[] =
	{
		{ "childShape", &hkpSingleShapeContainerClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "childShapeSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpConvexTransformShapeBaseClass(
		"hkpConvexTransformShapeBase",
		&hkpConvexShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConvexTransformShapeBaseClass_Members),
		HK_COUNT_OF(hkpConvexTransformShapeBaseClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBoxShapeClass_Members[] =
	{
		{ "halfExtents", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpBoxShapeClass(
		"hkpBoxShape",
		&hkpConvexShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBoxShapeClass_Members),
		HK_COUNT_OF(hkpBoxShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpCapsuleShapeRayHitTypeEnumItems[] =
	{
		{0, "HIT_CAP0"},
		{1, "HIT_CAP1"},
		{2, "HIT_BODY"},
	};
	static const hkInternalClassEnum hkpCapsuleShapeEnums[] = {
		{"RayHitType", hkpCapsuleShapeRayHitTypeEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkpCapsuleShapeRayHitTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpCapsuleShapeEnums[0]);
	static hkInternalClassMember hkpCapsuleShapeClass_Members[] =
	{
		{ "vertexA", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vertexB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpCapsuleShapeClass(
		"hkpCapsuleShape",
		&hkpConvexShapeClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpCapsuleShapeEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpCapsuleShapeClass_Members),
		HK_COUNT_OF(hkpCapsuleShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConvexTransformShapeClass_Members[] =
	{
		{ "transform", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConvexTransformShapeClass(
		"hkpConvexTransformShape",
		&hkpConvexTransformShapeBaseClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConvexTransformShapeClass_Members),
		HK_COUNT_OF(hkpConvexTransformShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConvexTranslateShapeClass_Members[] =
	{
		{ "translation", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConvexTranslateShapeClass(
		"hkpConvexTranslateShape",
		&hkpConvexTransformShapeBaseClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConvexTranslateShapeClass_Members),
		HK_COUNT_OF(hkpConvexTranslateShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConvexVerticesConnectivityClass_Members[] =
	{
		{ "vertexIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "numVerticesPerFace", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConvexVerticesConnectivityClass(
		"hkpConvexVerticesConnectivity",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConvexVerticesConnectivityClass_Members),
		HK_COUNT_OF(hkpConvexVerticesConnectivityClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConvexVerticesShape_FourVectorsClass_Members[] =
	{
		{ "x", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "y", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "z", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConvexVerticesShapeFourVectorsClass(
		"hkpConvexVerticesShapeFourVectors",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConvexVerticesShape_FourVectorsClass_Members),
		HK_COUNT_OF(hkpConvexVerticesShape_FourVectorsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConvexVerticesShapeClass_Members[] =
	{
		{ "aabbHalfExtents", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aabbCenter", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotatedVertices", &hkpConvexVerticesShapeFourVectorsClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "numVertices", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "planeEquations", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL },
		{ "connectivity", &hkpConvexVerticesConnectivityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConvexVerticesShapeClass(
		"hkpConvexVerticesShape",
		&hkpConvexShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConvexVerticesShapeClass_Members),
		HK_COUNT_OF(hkpConvexVerticesShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpCylinderShapeVertexIdEncodingEnumItems[] =
	{
		{7, "VERTEX_ID_ENCODING_IS_BASE_A_SHIFT"},
		{6, "VERTEX_ID_ENCODING_SIN_SIGN_SHIFT"},
		{5, "VERTEX_ID_ENCODING_COS_SIGN_SHIFT"},
		{4, "VERTEX_ID_ENCODING_IS_SIN_LESSER_SHIFT"},
		{15/*0x0f*/, "VERTEX_ID_ENCODING_VALUE_MASK"},
	};
	static const hkInternalClassEnum hkpCylinderShapeEnums[] = {
		{"VertexIdEncoding", hkpCylinderShapeVertexIdEncodingEnumItems, 5, HK_NULL, 0 }
	};
	const hkClassEnum* hkpCylinderShapeVertexIdEncodingEnum = reinterpret_cast<const hkClassEnum*>(&hkpCylinderShapeEnums[0]);
	static hkInternalClassMember hkpCylinderShapeClass_Members[] =
	{
		{ "cylRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "cylBaseRadiusFactorForHeightFieldCollisions", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vertexA", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vertexB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "perpendicular1", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "perpendicular2", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpCylinderShape_DefaultStruct
		{
			int s_defaultOffsets[6];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_cylBaseRadiusFactorForHeightFieldCollisions;
		};
		const hkpCylinderShape_DefaultStruct hkpCylinderShape_Default =
		{
			{-1,HK_OFFSET_OF(hkpCylinderShape_DefaultStruct,m_cylBaseRadiusFactorForHeightFieldCollisions),-1,-1,-1,-1},
			0.8f
		};
	}
	hkClass hkpCylinderShapeClass(
		"hkpCylinderShape",
		&hkpConvexShapeClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpCylinderShapeEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpCylinderShapeClass_Members),
		HK_COUNT_OF(hkpCylinderShapeClass_Members),
		&hkpCylinderShape_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSphereShapeClass_Members[] =
	{
		{ "pad16", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 3, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpSphereShapeClass(
		"hkpSphereShape",
		&hkpConvexShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSphereShapeClass_Members),
		HK_COUNT_OF(hkpSphereShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpTriangleShapeClass_Members[] =
	{
		{ "weldingInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "weldingType", HK_NULL, hkpWeldingUtilityWeldingTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "isExtruded", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vertexA", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vertexB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "vertexC", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extrusion", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpTriangleShape_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint8 /* hkEnum<hkpWeldingUtility::WeldingType, hkUint8> */ m_weldingType;
		};
		const hkpTriangleShape_DefaultStruct hkpTriangleShape_Default =
		{
			{-1,HK_OFFSET_OF(hkpTriangleShape_DefaultStruct,m_weldingType),-1,-1,-1,-1,-1},
			6
		};
	}
	hkClass hkpTriangleShapeClass(
		"hkpTriangleShape",
		&hkpConvexShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpTriangleShapeClass_Members),
		HK_COUNT_OF(hkpTriangleShapeClass_Members),
		&hkpTriangleShape_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConvexPieceMeshShapeClass_Members[] =
	{
		{ "convexPieceStream", &hkpConvexPieceStreamDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "displayMesh", &hkpShapeCollectionClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConvexPieceMeshShapeClass(
		"hkpConvexPieceMeshShape",
		&hkpShapeCollectionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConvexPieceMeshShapeClass_Members),
		HK_COUNT_OF(hkpConvexPieceMeshShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpFastMeshShapeClass(
		"hkpFastMeshShape",
		&hkpMeshShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpMeshShapeIndexStridingTypeEnumItems[] =
	{
		{0, "INDICES_INVALID"},
		{1, "INDICES_INT16"},
		{2, "INDICES_INT32"},
		{3, "INDICES_MAX_ID"},
	};
	static const hkInternalClassEnumItem hkpMeshShapeMaterialIndexStridingTypeEnumItems[] =
	{
		{0, "MATERIAL_INDICES_INVALID"},
		{1, "MATERIAL_INDICES_INT8"},
		{2, "MATERIAL_INDICES_INT16"},
		{3, "MATERIAL_INDICES_MAX_ID"},
	};
	static const hkInternalClassEnum hkpMeshShapeEnums[] = {
		{"IndexStridingType", hkpMeshShapeIndexStridingTypeEnumItems, 4, HK_NULL, 0 },
		{"MaterialIndexStridingType", hkpMeshShapeMaterialIndexStridingTypeEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkpMeshShapeIndexStridingTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpMeshShapeEnums[0]);
	const hkClassEnum* hkpMeshShapeMaterialIndexStridingTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpMeshShapeEnums[1]);
	static hkInternalClassMember hkpMeshShape_SubpartClass_Members[] =
	{
		{ "vertexBase", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "vertexStriding", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numVertices", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "indexBase", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "stridingType", HK_NULL, hkpMeshShapeIndexStridingTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "materialIndexStridingType", HK_NULL, hkpMeshShapeIndexStridingTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "indexStriding", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "flipAlternateTriangles", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numTriangles", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "materialIndexBase", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "materialIndexStriding", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "materialBase", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "materialStriding", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numMaterials", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "triangleOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpMeshShapeSubpart_DefaultStruct
		{
			int s_defaultOffsets[15];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			int m_triangleOffset;
		};
		const hkpMeshShapeSubpart_DefaultStruct hkpMeshShapeSubpart_Default =
		{
			{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkpMeshShapeSubpart_DefaultStruct,m_triangleOffset)},
			-1
		};
	}
	hkClass hkpMeshShapeSubpartClass(
		"hkpMeshShapeSubpart",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMeshShape_SubpartClass_Members),
		HK_COUNT_OF(hkpMeshShape_SubpartClass_Members),
		&hkpMeshShapeSubpart_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMeshShapeClass_Members[] =
	{
		{ "scaling", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numBitsForSubpartIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "subparts", &hkpMeshShapeSubpartClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "weldingInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "weldingType", HK_NULL, hkpWeldingUtilityWeldingTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pad", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 3, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpMeshShape_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint8 /* hkEnum<hkpWeldingUtility::WeldingType, hkUint8> */ m_weldingType;
		};
		const hkpMeshShape_DefaultStruct hkpMeshShape_Default =
		{
			{-1,-1,-1,-1,HK_OFFSET_OF(hkpMeshShape_DefaultStruct,m_weldingType),-1,-1},
			6
		};
	}
	hkClass hkpMeshShapeClass(
		"hkpMeshShape",
		&hkpShapeCollectionClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpMeshShapeEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkpMeshShapeClass_Members),
		HK_COUNT_OF(hkpMeshShapeClass_Members),
		&hkpMeshShape_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMultiSphereShapeClass_Members[] =
	{
		{ "numSpheres", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "spheres", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 8, 0, 0, HK_NULL }
	};
	hkClass hkpMultiSphereShapeClass(
		"hkpMultiSphereShape",
		&hkpSphereRepShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMultiSphereShapeClass_Members),
		HK_COUNT_OF(hkpMultiSphereShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStorageMeshShape_SubpartStorageClass_Members[] =
	{
		{ "vertices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "indices16", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "indices32", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "materialIndices", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "materials", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "materialIndices16", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL }
	};
	hkClass hkpStorageMeshShapeSubpartStorageClass(
		"hkpStorageMeshShapeSubpartStorage",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStorageMeshShape_SubpartStorageClass_Members),
		HK_COUNT_OF(hkpStorageMeshShape_SubpartStorageClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStorageMeshShapeClass_Members[] =
	{
		{ "storage", &hkpStorageMeshShapeSubpartStorageClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpStorageMeshShapeClass(
		"hkpStorageMeshShape",
		&hkpMeshShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStorageMeshShapeClass_Members),
		HK_COUNT_OF(hkpStorageMeshShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpHeightFieldShapeClass(
		"hkpHeightFieldShape",
		&hkpShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpSphereRepShapeClass(
		"hkpSphereRepShape",
		&hkpShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpCompressedSampledHeightFieldShapeClass_Members[] =
	{
		{ "storage", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "triangleFlip", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "offset", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "scale", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpCompressedSampledHeightFieldShapeClass(
		"hkpCompressedSampledHeightFieldShape",
		&hkpSampledHeightFieldShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpCompressedSampledHeightFieldShapeClass_Members),
		HK_COUNT_OF(hkpCompressedSampledHeightFieldShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPlaneShapeClass_Members[] =
	{
		{ "plane", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aabbCenter", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aabbHalfExtents", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPlaneShapeClass(
		"hkpPlaneShape",
		&hkpHeightFieldShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPlaneShapeClass_Members),
		HK_COUNT_OF(hkpPlaneShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpSampledHeightFieldShapeHeightFieldTypeEnumItems[] =
	{
		{0, "HEIGHTFIELD_STORAGE"},
		{1, "HEIGHTFIELD_COMPRESSED"},
		{2, "HEIGHTFIELD_USER"},
		{3, "HEIGHTFIELD_MAX_ID"},
	};
	static const hkInternalClassEnum hkpSampledHeightFieldShapeEnums[] = {
		{"HeightFieldType", hkpSampledHeightFieldShapeHeightFieldTypeEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkpSampledHeightFieldShapeHeightFieldTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpSampledHeightFieldShapeEnums[0]);
	static hkInternalClassMember hkpSampledHeightFieldShapeClass_Members[] =
	{
		{ "xRes", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "zRes", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "heightCenter", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useProjectionBasedHeight", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "heightfieldType", HK_NULL, hkpSampledHeightFieldShapeHeightFieldTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "intToFloatScale", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "floatToIntScale", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "floatToIntOffsetFloorCorrected", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extents", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpSampledHeightFieldShape_DefaultStruct
		{
			int s_defaultOffsets[9];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint8 /* hkEnum<HeightFieldType, hkUint8> */ m_heightfieldType;
		};
		const hkpSampledHeightFieldShape_DefaultStruct hkpSampledHeightFieldShape_Default =
		{
			{-1,-1,-1,-1,HK_OFFSET_OF(hkpSampledHeightFieldShape_DefaultStruct,m_heightfieldType),-1,-1,-1,-1},
			2
		};
	}
	hkClass hkpSampledHeightFieldShapeClass(
		"hkpSampledHeightFieldShape",
		&hkpHeightFieldShapeClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpSampledHeightFieldShapeEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpSampledHeightFieldShapeClass_Members),
		HK_COUNT_OF(hkpSampledHeightFieldShapeClass_Members),
		&hkpSampledHeightFieldShape_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStorageSampledHeightFieldShapeClass_Members[] =
	{
		{ "storage", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "triangleFlip", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpStorageSampledHeightFieldShapeClass(
		"hkpStorageSampledHeightFieldShape",
		&hkpSampledHeightFieldShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStorageSampledHeightFieldShapeClass_Members),
		HK_COUNT_OF(hkpStorageSampledHeightFieldShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpTriSampledHeightFieldBvTreeShapeClass_Members[] =
	{
		{ "childContainer", &hkpSingleShapeContainerClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "childSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "wantAabbRejectionTest", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "padding", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 12, 0, 0, HK_NULL }
	};
	hkClass hkpTriSampledHeightFieldBvTreeShapeClass(
		"hkpTriSampledHeightFieldBvTreeShape",
		&hkpBvTreeShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpTriSampledHeightFieldBvTreeShapeClass_Members),
		HK_COUNT_OF(hkpTriSampledHeightFieldBvTreeShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpTriSampledHeightFieldCollectionClass_Members[] =
	{
		{ "heightfield", &hkpSampledHeightFieldShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "childSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "weldingInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL },
		{ "triangleExtrusion", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpTriSampledHeightFieldCollectionClass(
		"hkpTriSampledHeightFieldCollection",
		&hkpShapeCollectionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpTriSampledHeightFieldCollectionClass_Members),
		HK_COUNT_OF(hkpTriSampledHeightFieldCollectionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBvShapeClass_Members[] =
	{
		{ "boundingVolumeShape", &hkpShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "childShape", &hkpSingleShapeContainerClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpBvShapeClass(
		"hkpBvShape",
		&hkpShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBvShapeClass_Members),
		HK_COUNT_OF(hkpBvShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConvexListShapeClass_Members[] =
	{
		{ "minDistanceToUseConvexHullForGetClosestPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aabbHalfExtents", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aabbCenter", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useCachedAabb", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "childShapes", &hkpConvexShapeClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConvexListShapeClass(
		"hkpConvexListShape",
		&hkpConvexShapeClass,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConvexListShapeClass_Members),
		HK_COUNT_OF(hkpConvexListShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMultiRayShape_RayClass_Members[] =
	{
		{ "start", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "end", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMultiRayShapeRayClass(
		"hkpMultiRayShapeRay",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMultiRayShape_RayClass_Members),
		HK_COUNT_OF(hkpMultiRayShape_RayClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMultiRayShapeClass_Members[] =
	{
		{ "rays", &hkpMultiRayShapeRayClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "rayPenetrationDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMultiRayShapeClass(
		"hkpMultiRayShape",
		&hkpShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMultiRayShapeClass_Members),
		HK_COUNT_OF(hkpMultiRayShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpPhantomCallbackShapeClass(
		"hkpPhantomCallbackShape",
		&hkpShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpTransformShapeClass_Members[] =
	{
		{ "childShape", &hkpSingleShapeContainerClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "childShapeSize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "rotation", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transform", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpTransformShapeClass(
		"hkpTransformShape",
		&hkpShapeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpTransformShapeClass_Members),
		HK_COUNT_OF(hkpTransformShapeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpRayShapeCollectionFilterClass(
		"hkpRayShapeCollectionFilter",
		HK_NULL,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpShapeRayCastInputClass_Members[] =
	{
		{ "from", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "to", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "filterInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rayShapeCollectionFilter", &hkpRayShapeCollectionFilterClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpShapeRayCastInputClass(
		"hkpShapeRayCastInput",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpShapeRayCastInputClass_Members),
		HK_COUNT_OF(hkpShapeRayCastInputClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpShapeInfoClass_Members[] =
	{
		{ "shape", &hkpShapeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "isHierarchicalCompound", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hkdShapesCollected", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "childShapeNames", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_CSTRING, 0, 0, 0, HK_NULL },
		{ "childTransforms", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_TRANSFORM, 0, 0, 0, HK_NULL },
		{ "transform", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpShapeInfoClass(
		"hkpShapeInfo",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpShapeInfoClass_Members),
		HK_COUNT_OF(hkpShapeInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpWeldingUtilityWeldingTypeEnumItems[] =
	{
		{0, "WELDING_TYPE_ANTICLOCKWISE"},
		{4, "WELDING_TYPE_CLOCKWISE"},
		{5, "WELDING_TYPE_TWO_SIDED"},
		{6, "WELDING_TYPE_NONE"},
	};
	static const hkInternalClassEnumItem hkpWeldingUtilitySectorTypeEnumItems[] =
	{
		{1, "ACCEPT_0"},
		{0, "SNAP_0"},
		{2, "REJECT"},
		{4, "SNAP_1"},
		{3, "ACCEPT_1"},
	};
	static const hkInternalClassEnumItem hkpWeldingUtilityNumAnglesEnumItems[] =
	{
		{31, "NUM_ANGLES"},
	};
	static const hkInternalClassEnum hkpWeldingUtilityEnums[] = {
		{"WeldingType", hkpWeldingUtilityWeldingTypeEnumItems, 4, HK_NULL, 0 },
		{"SectorType", hkpWeldingUtilitySectorTypeEnumItems, 5, HK_NULL, 0 },
		{"NumAngles", hkpWeldingUtilityNumAnglesEnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hkpWeldingUtilityWeldingTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpWeldingUtilityEnums[0]);
	const hkClassEnum* hkpWeldingUtilitySectorTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpWeldingUtilityEnums[1]);
	const hkClassEnum* hkpWeldingUtilityNumAnglesEnum = reinterpret_cast<const hkClassEnum*>(&hkpWeldingUtilityEnums[2]);
	hkClass hkpWeldingUtilityClass(
		"hkpWeldingUtility",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpWeldingUtilityEnums),
		3,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpConstraintAtomAtomTypeEnumItems[] =
	{
		{0, "TYPE_INVALID"},
		{1, "TYPE_BRIDGE"},
		{2, "TYPE_SET_LOCAL_TRANSFORMS"},
		{3, "TYPE_SET_LOCAL_TRANSLATIONS"},
		{4, "TYPE_SET_LOCAL_ROTATIONS"},
		{5, "TYPE_BALL_SOCKET"},
		{6, "TYPE_STIFF_SPRING"},
		{7, "TYPE_LIN"},
		{8, "TYPE_LIN_SOFT"},
		{9, "TYPE_LIN_LIMIT"},
		{10, "TYPE_LIN_FRICTION"},
		{11, "TYPE_LIN_MOTOR"},
		{12, "TYPE_2D_ANG"},
		{13, "TYPE_ANG"},
		{14, "TYPE_ANG_LIMIT"},
		{15, "TYPE_TWIST_LIMIT"},
		{16, "TYPE_CONE_LIMIT"},
		{17, "TYPE_ANG_FRICTION"},
		{18, "TYPE_ANG_MOTOR"},
		{19, "TYPE_RAGDOLL_MOTOR"},
		{20, "TYPE_PULLEY"},
		{21, "TYPE_OVERWRITE_PIVOT"},
		{22, "TYPE_CONTACT"},
		{23, "TYPE_MODIFIER_SOFT_CONTACT"},
		{24, "TYPE_MODIFIER_MASS_CHANGER"},
		{25, "TYPE_MODIFIER_VISCOUS_SURFACE"},
		{26, "TYPE_MODIFIER_MOVING_SURFACE"},
		{27, "TYPE_MODIFIER_IGNORE_CONSTRAINT"},
		{28, "TYPE_MAX"},
	};
	static const hkInternalClassEnumItem hkpConstraintAtomCallbackRequestEnumItems[] =
	{
		{0, "CALLBACK_REQUEST_NONE"},
		{1, "CALLBACK_REQUEST_NEW_CONTACT_POINT"},
		{2, "CALLBACK_REQUEST_SETUP_PPU_ONLY"},
		{4, "CALLBACK_REQUEST_SETUP_CALLBACK"},
	};
	static const hkInternalClassEnum hkpConstraintAtomEnums[] = {
		{"AtomType", hkpConstraintAtomAtomTypeEnumItems, 29, HK_NULL, 0 },
		{"CallbackRequest", hkpConstraintAtomCallbackRequestEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkpConstraintAtomAtomTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintAtomEnums[0]);
	const hkClassEnum* hkpConstraintAtomCallbackRequestEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintAtomEnums[1]);
	static hkInternalClassMember hkpConstraintAtomClass_Members[] =
	{
		{ "type", HK_NULL, hkpConstraintAtomAtomTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT16, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConstraintAtomClass(
		"hkpConstraintAtom",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpConstraintAtomEnums),
		2,
		reinterpret_cast<const hkClassMember*>(hkpConstraintAtomClass_Members),
		HK_COUNT_OF(hkpConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBridgeConstraintAtomClass_Members[] =
	{
		{ "buildJacobianFunc", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "constraintData", &hkpConstraintDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpBridgeConstraintAtomClass(
		"hkpBridgeConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBridgeConstraintAtomClass_Members),
		HK_COUNT_OF(hkpBridgeConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBridgeAtomsClass_Members[] =
	{
		{ "bridgeAtom", &hkpBridgeConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpBridgeAtomsClass(
		"hkpBridgeAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBridgeAtomsClass_Members),
		HK_COUNT_OF(hkpBridgeAtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSimpleContactConstraintAtomClass_Members[] =
	{
		{ "sizeOfAllAtoms", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numContactPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numReservedContactPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numUserDatasForBodyA", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numUserDatasForBodyB", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contactPointPropertiesStriding", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxNumContactPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "info", &hkpSimpleContactConstraintDataInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpSimpleContactConstraintAtomClass(
		"hkpSimpleContactConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSimpleContactConstraintAtomClass_Members),
		HK_COUNT_OF(hkpSimpleContactConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBallSocketConstraintAtomClass_Members[] =
	{
		{ "maxImpulse", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bodiesToNotify", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpBallSocketConstraintAtom_DefaultStruct
		{
			int s_defaultOffsets[2];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_maxImpulse;
		};
		const hkpBallSocketConstraintAtom_DefaultStruct hkpBallSocketConstraintAtom_Default =
		{
			{HK_OFFSET_OF(hkpBallSocketConstraintAtom_DefaultStruct,m_maxImpulse),-1},
			HK_REAL_MAX
		};
	}
	hkClass hkpBallSocketConstraintAtomClass(
		"hkpBallSocketConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBallSocketConstraintAtomClass_Members),
		HK_COUNT_OF(hkpBallSocketConstraintAtomClass_Members),
		&hkpBallSocketConstraintAtom_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStiffSpringConstraintAtomClass_Members[] =
	{
		{ "length", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpStiffSpringConstraintAtomClass(
		"hkpStiffSpringConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStiffSpringConstraintAtomClass_Members),
		HK_COUNT_OF(hkpStiffSpringConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSetLocalTransformsConstraintAtomClass_Members[] =
	{
		{ "transformA", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "transformB", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSetLocalTransformsConstraintAtomClass(
		"hkpSetLocalTransformsConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSetLocalTransformsConstraintAtomClass_Members),
		HK_COUNT_OF(hkpSetLocalTransformsConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSetLocalTranslationsConstraintAtomClass_Members[] =
	{
		{ "translationA", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "translationB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSetLocalTranslationsConstraintAtomClass(
		"hkpSetLocalTranslationsConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSetLocalTranslationsConstraintAtomClass_Members),
		HK_COUNT_OF(hkpSetLocalTranslationsConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSetLocalRotationsConstraintAtomClass_Members[] =
	{
		{ "rotationA", HK_NULL, HK_NULL, hkClassMember::TYPE_ROTATION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rotationB", HK_NULL, HK_NULL, hkClassMember::TYPE_ROTATION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSetLocalRotationsConstraintAtomClass(
		"hkpSetLocalRotationsConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSetLocalRotationsConstraintAtomClass_Members),
		HK_COUNT_OF(hkpSetLocalRotationsConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpOverwritePivotConstraintAtomClass_Members[] =
	{
		{ "copyToPivotBFromPivotA", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpOverwritePivotConstraintAtomClass(
		"hkpOverwritePivotConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpOverwritePivotConstraintAtomClass_Members),
		HK_COUNT_OF(hkpOverwritePivotConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpLinConstraintAtomClass_Members[] =
	{
		{ "axisIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpLinConstraintAtomClass(
		"hkpLinConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpLinConstraintAtomClass_Members),
		HK_COUNT_OF(hkpLinConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpLinSoftConstraintAtomClass_Members[] =
	{
		{ "axisIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tau", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpLinSoftConstraintAtomClass(
		"hkpLinSoftConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpLinSoftConstraintAtomClass_Members),
		HK_COUNT_OF(hkpLinSoftConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpLinLimitConstraintAtomClass_Members[] =
	{
		{ "axisIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "min", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "max", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpLinLimitConstraintAtomClass(
		"hkpLinLimitConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpLinLimitConstraintAtomClass_Members),
		HK_COUNT_OF(hkpLinLimitConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkp2dAngConstraintAtomClass_Members[] =
	{
		{ "freeRotationAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkp2dAngConstraintAtomClass(
		"hkp2dAngConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkp2dAngConstraintAtomClass_Members),
		HK_COUNT_OF(hkp2dAngConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpAngConstraintAtomClass_Members[] =
	{
		{ "firstConstrainedAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numConstrainedAxes", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpAngConstraintAtomClass(
		"hkpAngConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpAngConstraintAtomClass_Members),
		HK_COUNT_OF(hkpAngConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpAngLimitConstraintAtomClass_Members[] =
	{
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "limitAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angularLimitsTauFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpAngLimitConstraintAtom_DefaultStruct
		{
			int s_defaultOffsets[5];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_angularLimitsTauFactor;
		};
		const hkpAngLimitConstraintAtom_DefaultStruct hkpAngLimitConstraintAtom_Default =
		{
			{-1,-1,-1,-1,HK_OFFSET_OF(hkpAngLimitConstraintAtom_DefaultStruct,m_angularLimitsTauFactor)},
			1.0
		};
	}
	hkClass hkpAngLimitConstraintAtomClass(
		"hkpAngLimitConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpAngLimitConstraintAtomClass_Members),
		HK_COUNT_OF(hkpAngLimitConstraintAtomClass_Members),
		&hkpAngLimitConstraintAtom_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpTwistLimitConstraintAtomClass_Members[] =
	{
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "twistAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "refAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angularLimitsTauFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpTwistLimitConstraintAtom_DefaultStruct
		{
			int s_defaultOffsets[6];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_angularLimitsTauFactor;
		};
		const hkpTwistLimitConstraintAtom_DefaultStruct hkpTwistLimitConstraintAtom_Default =
		{
			{-1,-1,-1,-1,-1,HK_OFFSET_OF(hkpTwistLimitConstraintAtom_DefaultStruct,m_angularLimitsTauFactor)},
			1.0
		};
	}
	hkClass hkpTwistLimitConstraintAtomClass(
		"hkpTwistLimitConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpTwistLimitConstraintAtomClass_Members),
		HK_COUNT_OF(hkpTwistLimitConstraintAtomClass_Members),
		&hkpTwistLimitConstraintAtom_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpConeLimitConstraintAtomMeasurementModeEnumItems[] =
	{
		{0, "ZERO_WHEN_VECTORS_ALIGNED"},
		{1, "ZERO_WHEN_VECTORS_PERPENDICULAR"},
	};
	static const hkInternalClassEnum hkpConeLimitConstraintAtomEnums[] = {
		{"MeasurementMode", hkpConeLimitConstraintAtomMeasurementModeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkpConeLimitConstraintAtomMeasurementModeEnum = reinterpret_cast<const hkClassEnum*>(&hkpConeLimitConstraintAtomEnums[0]);
	static hkInternalClassMember hkpConeLimitConstraintAtomClass_Members[] =
	{
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "twistAxisInA", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "refAxisInB", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angleMeasurementMode", HK_NULL, hkpConeLimitConstraintAtomMeasurementModeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "memOffsetToAngleOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angularLimitsTauFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpConeLimitConstraintAtom_DefaultStruct
		{
			int s_defaultOffsets[8];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkUint8 m_memOffsetToAngleOffset;
			hkReal m_angularLimitsTauFactor;
		};
		const hkpConeLimitConstraintAtom_DefaultStruct hkpConeLimitConstraintAtom_Default =
		{
			{-1,-1,-1,-1,HK_OFFSET_OF(hkpConeLimitConstraintAtom_DefaultStruct,m_memOffsetToAngleOffset),-1,-1,HK_OFFSET_OF(hkpConeLimitConstraintAtom_DefaultStruct,m_angularLimitsTauFactor)},
			1,1.0
		};
	}
	hkClass hkpConeLimitConstraintAtomClass(
		"hkpConeLimitConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpConeLimitConstraintAtomEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpConeLimitConstraintAtomClass_Members),
		HK_COUNT_OF(hkpConeLimitConstraintAtomClass_Members),
		&hkpConeLimitConstraintAtom_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpAngFrictionConstraintAtomClass_Members[] =
	{
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "firstFrictionAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numFrictionAxes", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxFrictionTorque", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpAngFrictionConstraintAtomClass(
		"hkpAngFrictionConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpAngFrictionConstraintAtomClass_Members),
		HK_COUNT_OF(hkpAngFrictionConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpAngMotorConstraintAtomClass_Members[] =
	{
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "motorAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initializedOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "previousTargetAngleOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "correspondingAngLimitSolverResultOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "motor", &hkpConstraintMotorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpAngMotorConstraintAtomClass(
		"hkpAngMotorConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpAngMotorConstraintAtomClass_Members),
		HK_COUNT_OF(hkpAngMotorConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpRagdollMotorConstraintAtomClass_Members[] =
	{
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initializedOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "previousTargetAnglesOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "target_bRca", HK_NULL, HK_NULL, hkClassMember::TYPE_MATRIX3, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "motors", &hkpConstraintMotorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 3, 0, 0, HK_NULL }
	};
	hkClass hkpRagdollMotorConstraintAtomClass(
		"hkpRagdollMotorConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpRagdollMotorConstraintAtomClass_Members),
		HK_COUNT_OF(hkpRagdollMotorConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpLinFrictionConstraintAtomClass_Members[] =
	{
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "frictionAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxFrictionForce", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpLinFrictionConstraintAtomClass(
		"hkpLinFrictionConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpLinFrictionConstraintAtomClass_Members),
		HK_COUNT_OF(hkpLinFrictionConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpLinMotorConstraintAtomClass_Members[] =
	{
		{ "isEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "motorAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initializedOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "previousTargetPositionOffset", HK_NULL, HK_NULL, hkClassMember::TYPE_INT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "targetPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "motor", &hkpConstraintMotorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpLinMotorConstraintAtomClass(
		"hkpLinMotorConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpLinMotorConstraintAtomClass_Members),
		HK_COUNT_OF(hkpLinMotorConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPulleyConstraintAtomClass_Members[] =
	{
		{ "fixedPivotAinWorld", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "fixedPivotBinWorld", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ropeLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "leverageOnBodyB", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPulleyConstraintAtomClass(
		"hkpPulleyConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPulleyConstraintAtomClass_Members),
		HK_COUNT_OF(hkpPulleyConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpModifierConstraintAtomClass_Members[] =
	{
		{ "modifierAtomSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL },
		{ "childSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "child", &hkpConstraintAtomClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "pad", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL }
	};
	hkClass hkpModifierConstraintAtomClass(
		"hkpModifierConstraintAtom",
		&hkpConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpModifierConstraintAtomClass_Members),
		HK_COUNT_OF(hkpModifierConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSoftContactModifierConstraintAtomClass_Members[] =
	{
		{ "tau", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxAcceleration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSoftContactModifierConstraintAtomClass(
		"hkpSoftContactModifierConstraintAtom",
		&hkpModifierConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSoftContactModifierConstraintAtomClass_Members),
		HK_COUNT_OF(hkpSoftContactModifierConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMassChangerModifierConstraintAtomClass_Members[] =
	{
		{ "factorA", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "factorB", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pad16", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 2, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpMassChangerModifierConstraintAtomClass(
		"hkpMassChangerModifierConstraintAtom",
		&hkpModifierConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMassChangerModifierConstraintAtomClass_Members),
		HK_COUNT_OF(hkpMassChangerModifierConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpViscousSurfaceModifierConstraintAtomClass(
		"hkpViscousSurfaceModifierConstraintAtom",
		&hkpModifierConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMovingSurfaceModifierConstraintAtomClass_Members[] =
	{
		{ "velocity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMovingSurfaceModifierConstraintAtomClass(
		"hkpMovingSurfaceModifierConstraintAtom",
		&hkpModifierConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMovingSurfaceModifierConstraintAtomClass_Members),
		HK_COUNT_OF(hkpMovingSurfaceModifierConstraintAtomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpIgnoreModifierConstraintAtomClass(
		"hkpIgnoreModifierConstraintAtom",
		&hkpModifierConstraintAtomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSimpleContactConstraintDataInfoClass_Members[] =
	{
		{ "flags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL },
		{ "index", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 7, 0, 0, HK_NULL }
	};
	hkClass hkpSimpleContactConstraintDataInfoClass(
		"hkpSimpleContactConstraintDataInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSimpleContactConstraintDataInfoClass_Members),
		HK_COUNT_OF(hkpSimpleContactConstraintDataInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpActionClass_Members[] =
	{
		{ "world", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "island", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "userData", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpActionClass(
		"hkpAction",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpActionClass_Members),
		HK_COUNT_OF(hkpActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpArrayActionClass_Members[] =
	{
		{ "entities", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpArrayActionClass(
		"hkpArrayAction",
		&hkpActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpArrayActionClass_Members),
		HK_COUNT_OF(hkpArrayActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBinaryActionClass_Members[] =
	{
		{ "entityA", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "entityB", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpBinaryActionClass(
		"hkpBinaryAction",
		&hkpActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBinaryActionClass_Members),
		HK_COUNT_OF(hkpBinaryActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpUnaryActionClass_Members[] =
	{
		{ "entity", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpUnaryActionClass(
		"hkpUnaryAction",
		&hkpActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpUnaryActionClass_Members),
		HK_COUNT_OF(hkpUnaryActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpMaterialResponseTypeEnumItems[] =
	{
		{0, "RESPONSE_INVALID"},
		{1, "RESPONSE_SIMPLE_CONTACT"},
		{2, "RESPONSE_REPORTING"},
		{3, "RESPONSE_NONE"},
		{4, "RESPONSE_MAX_ID"},
	};
	static const hkInternalClassEnum hkpMaterialEnums[] = {
		{"ResponseType", hkpMaterialResponseTypeEnumItems, 5, HK_NULL, 0 }
	};
	const hkClassEnum* hkpMaterialResponseTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpMaterialEnums[0]);
	static hkInternalClassMember hkpMaterialClass_Members[] =
	{
		{ "responseType", HK_NULL, hkpMaterialResponseTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "friction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "restitution", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMaterialClass(
		"hkpMaterial",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpMaterialEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpMaterialClass_Members),
		HK_COUNT_OF(hkpMaterialClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPropertyValueClass_Members[] =
	{
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT64, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPropertyValueClass(
		"hkpPropertyValue",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPropertyValueClass_Members),
		HK_COUNT_OF(hkpPropertyValueClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPropertyClass_Members[] =
	{
		{ "key", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "alignmentPadding", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "value", &hkpPropertyValueClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPropertyClass(
		"hkpProperty",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPropertyClass_Members),
		HK_COUNT_OF(hkpPropertyClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpConstraintDataConstraintTypeEnumItems[] =
	{
		{0, "CONSTRAINT_TYPE_BALLANDSOCKET"},
		{1, "CONSTRAINT_TYPE_HINGE"},
		{2, "CONSTRAINT_TYPE_LIMITEDHINGE"},
		{3, "CONSTRAINT_TYPE_POINTTOPATH"},
		{6, "CONSTRAINT_TYPE_PRISMATIC"},
		{7, "CONSTRAINT_TYPE_RAGDOLL"},
		{8, "CONSTRAINT_TYPE_STIFFSPRING"},
		{9, "CONSTRAINT_TYPE_WHEEL"},
		{10, "CONSTRAINT_TYPE_GENERIC"},
		{11, "CONSTRAINT_TYPE_CONTACT"},
		{12, "CONSTRAINT_TYPE_BREAKABLE"},
		{13, "CONSTRAINT_TYPE_MALLEABLE"},
		{14, "CONSTRAINT_TYPE_POINTTOPLANE"},
		{15, "CONSTRAINT_TYPE_PULLEY"},
		{16, "CONSTRAINT_TYPE_ROTATIONAL"},
		{18, "CONSTRAINT_TYPE_HINGE_LIMITS"},
		{19, "CONSTRAINT_TYPE_RAGDOLL_LIMITS"},
		{20, "CONSTRAINT_TYPE_CUSTOM"},
		{100, "BEGIN_CONSTRAINT_CHAIN_TYPES"},
		{100, "CONSTRAINT_TYPE_STIFF_SPRING_CHAIN"},
		{101, "CONSTRAINT_TYPE_BALL_SOCKET_CHAIN"},
		{102, "CONSTRAINT_TYPE_POWERED_CHAIN"},
	};
	static const hkInternalClassEnum hkpConstraintDataEnums[] = {
		{"ConstraintType", hkpConstraintDataConstraintTypeEnumItems, 22, HK_NULL, 0 }
	};
	const hkClassEnum* hkpConstraintDataConstraintTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintDataEnums[0]);
	static hkInternalClassMember hkpConstraintDataClass_Members[] =
	{
		{ "userData", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConstraintDataClass(
		"hkpConstraintData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpConstraintDataEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpConstraintDataClass_Members),
		HK_COUNT_OF(hkpConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpConstraintInstanceConstraintPriorityEnumItems[] =
	{
		{0, "PRIORITY_INVALID"},
		{1, "PRIORITY_PSI"},
		{2, "PRIORITY_TOI"},
		{3, "PRIORITY_TOI_HIGHER"},
		{4, "PRIORITY_TOI_FORCED"},
	};
	static const hkInternalClassEnumItem hkpConstraintInstanceInstanceTypeEnumItems[] =
	{
		{0, "TYPE_NORMAL"},
		{1, "TYPE_CHAIN"},
	};
	static const hkInternalClassEnumItem hkpConstraintInstanceAddReferencesEnumItems[] =
	{
		{0, "DO_NOT_ADD_REFERENCES"},
		{1, "DO_ADD_REFERENCES"},
	};
	static const hkInternalClassEnumItem hkpConstraintInstanceCloningModeEnumItems[] =
	{
		{0, "CLONE_INSTANCES_ONLY"},
		{1, "CLONE_DATAS_WITH_MOTORS"},
	};
	static const hkInternalClassEnum hkpConstraintInstanceEnums[] = {
		{"ConstraintPriority", hkpConstraintInstanceConstraintPriorityEnumItems, 5, HK_NULL, 0 },
		{"InstanceType", hkpConstraintInstanceInstanceTypeEnumItems, 2, HK_NULL, 0 },
		{"AddReferences", hkpConstraintInstanceAddReferencesEnumItems, 2, HK_NULL, 0 },
		{"CloningMode", hkpConstraintInstanceCloningModeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkpConstraintInstanceConstraintPriorityEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintInstanceEnums[0]);
	const hkClassEnum* hkpConstraintInstanceInstanceTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintInstanceEnums[1]);
	const hkClassEnum* hkpConstraintInstanceAddReferencesEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintInstanceEnums[2]);
	const hkClassEnum* hkpConstraintInstanceCloningModeEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintInstanceEnums[3]);
	static hkInternalClassMember hkpConstraintInstanceClass_Members[] =
	{
		{ "owner", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "data", &hkpConstraintDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "constraintModifiers", &hkpModifierConstraintAtomClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "entities", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 2, 0, 0, HK_NULL },
		{ "priority", HK_NULL, hkpConstraintInstanceConstraintPriorityEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "wantRuntime", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "userData", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "internal", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpConstraintInstanceClass(
		"hkpConstraintInstance",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpConstraintInstanceEnums),
		4,
		reinterpret_cast<const hkClassMember*>(hkpConstraintInstanceClass_Members),
		HK_COUNT_OF(hkpConstraintInstanceClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBallAndSocketConstraintData_AtomsClass_Members[] =
	{
		{ "pivots", &hkpSetLocalTranslationsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ballSocket", &hkpBallSocketConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpBallAndSocketConstraintDataAtomsClass(
		"hkpBallAndSocketConstraintDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBallAndSocketConstraintData_AtomsClass_Members),
		HK_COUNT_OF(hkpBallAndSocketConstraintData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBallAndSocketConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpBallAndSocketConstraintDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpBallAndSocketConstraintDataClass(
		"hkpBallAndSocketConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBallAndSocketConstraintDataClass_Members),
		HK_COUNT_OF(hkpBallAndSocketConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpHingeConstraintDataAtomsAxisEnumItems[] =
	{
		{0, "AXIS_AXLE"},
	};
	static const hkInternalClassEnum hkpHingeConstraintDataAtomsEnums[] = {
		{"Axis", hkpHingeConstraintDataAtomsAxisEnumItems, 1, HK_NULL, 0 }
	};
	const hkClassEnum* hkpHingeConstraintDataAtomsAxisEnum = reinterpret_cast<const hkClassEnum*>(&hkpHingeConstraintDataAtomsEnums[0]);
	static hkInternalClassMember hkpHingeConstraintData_AtomsClass_Members[] =
	{
		{ "transforms", &hkpSetLocalTransformsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "2dAng", &hkp2dAngConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ballSocket", &hkpBallSocketConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpHingeConstraintDataAtomsClass(
		"hkpHingeConstraintDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpHingeConstraintDataAtomsEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpHingeConstraintData_AtomsClass_Members),
		HK_COUNT_OF(hkpHingeConstraintData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpHingeConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpHingeConstraintDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpHingeConstraintDataClass(
		"hkpHingeConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpHingeConstraintDataClass_Members),
		HK_COUNT_OF(hkpHingeConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpLimitedHingeConstraintDataAtomsAxisEnumItems[] =
	{
		{0, "AXIS_AXLE"},
		{1, "AXIS_PERP_TO_AXLE_1"},
		{2, "AXIS_PERP_TO_AXLE_2"},
	};
	static const hkInternalClassEnum hkpLimitedHingeConstraintDataAtomsEnums[] = {
		{"Axis", hkpLimitedHingeConstraintDataAtomsAxisEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkpLimitedHingeConstraintDataAtomsAxisEnum = reinterpret_cast<const hkClassEnum*>(&hkpLimitedHingeConstraintDataAtomsEnums[0]);
	static hkInternalClassMember hkpLimitedHingeConstraintData_AtomsClass_Members[] =
	{
		{ "transforms", &hkpSetLocalTransformsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angMotor", &hkpAngMotorConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angFriction", &hkpAngFrictionConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angLimit", &hkpAngLimitConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "2dAng", &hkp2dAngConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ballSocket", &hkpBallSocketConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpLimitedHingeConstraintDataAtomsClass(
		"hkpLimitedHingeConstraintDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpLimitedHingeConstraintDataAtomsEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpLimitedHingeConstraintData_AtomsClass_Members),
		HK_COUNT_OF(hkpLimitedHingeConstraintData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpLimitedHingeConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpLimitedHingeConstraintDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpLimitedHingeConstraintDataClass(
		"hkpLimitedHingeConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpLimitedHingeConstraintDataClass_Members),
		HK_COUNT_OF(hkpLimitedHingeConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpLinearParametricCurveClass_Members[] =
	{
		{ "smoothingFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "closedLoop", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dirNotParallelToTangentAlongWholePath", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "points", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL },
		{ "distance", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL }
	};
	hkClass hkpLinearParametricCurveClass(
		"hkpLinearParametricCurve",
		&hkpParametricCurveClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpLinearParametricCurveClass_Members),
		HK_COUNT_OF(hkpLinearParametricCurveClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpParametricCurveClass(
		"hkpParametricCurve",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpPointToPathConstraintDataOrientationConstraintTypeEnumItems[] =
	{
		{0, "CONSTRAIN_ORIENTATION_INVALID"},
		{1, "CONSTRAIN_ORIENTATION_NONE"},
		{2, "CONSTRAIN_ORIENTATION_ALLOW_SPIN"},
		{3, "CONSTRAIN_ORIENTATION_TO_PATH"},
		{4, "CONSTRAIN_ORIENTATION_MAX_ID"},
	};
	static const hkInternalClassEnum hkpPointToPathConstraintDataEnums[] = {
		{"OrientationConstraintType", hkpPointToPathConstraintDataOrientationConstraintTypeEnumItems, 5, HK_NULL, 0 }
	};
	const hkClassEnum* hkpPointToPathConstraintDataOrientationConstraintTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpPointToPathConstraintDataEnums[0]);
	static hkInternalClassMember hkpPointToPathConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpBridgeAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "path", &hkpParametricCurveClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "maxFrictionForce", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angularConstrainedDOF", HK_NULL, hkpPointToPathConstraintDataOrientationConstraintTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "transform_OS_KS", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL }
	};
	hkClass hkpPointToPathConstraintDataClass(
		"hkpPointToPathConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpPointToPathConstraintDataEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpPointToPathConstraintDataClass_Members),
		HK_COUNT_OF(hkpPointToPathConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPointToPlaneConstraintData_AtomsClass_Members[] =
	{
		{ "transforms", &hkpSetLocalTransformsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lin", &hkpLinConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPointToPlaneConstraintDataAtomsClass(
		"hkpPointToPlaneConstraintDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPointToPlaneConstraintData_AtomsClass_Members),
		HK_COUNT_OF(hkpPointToPlaneConstraintData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPointToPlaneConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpPointToPlaneConstraintDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpPointToPlaneConstraintDataClass(
		"hkpPointToPlaneConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPointToPlaneConstraintDataClass_Members),
		HK_COUNT_OF(hkpPointToPlaneConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpPrismaticConstraintDataAtomsAxisEnumItems[] =
	{
		{0, "AXIS_SHAFT"},
		{1, "AXIS_PERP_TO_SHAFT"},
	};
	static const hkInternalClassEnum hkpPrismaticConstraintDataAtomsEnums[] = {
		{"Axis", hkpPrismaticConstraintDataAtomsAxisEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkpPrismaticConstraintDataAtomsAxisEnum = reinterpret_cast<const hkClassEnum*>(&hkpPrismaticConstraintDataAtomsEnums[0]);
	static hkInternalClassMember hkpPrismaticConstraintData_AtomsClass_Members[] =
	{
		{ "transforms", &hkpSetLocalTransformsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "motor", &hkpLinMotorConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "friction", &hkpLinFrictionConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ang", &hkpAngConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lin0", &hkpLinConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lin1", &hkpLinConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "linLimit", &hkpLinLimitConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPrismaticConstraintDataAtomsClass(
		"hkpPrismaticConstraintDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpPrismaticConstraintDataAtomsEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpPrismaticConstraintData_AtomsClass_Members),
		HK_COUNT_OF(hkpPrismaticConstraintData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPrismaticConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpPrismaticConstraintDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpPrismaticConstraintDataClass(
		"hkpPrismaticConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPrismaticConstraintDataClass_Members),
		HK_COUNT_OF(hkpPrismaticConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpRagdollConstraintDataMotorIndexEnumItems[] =
	{
		{0, "MOTOR_TWIST"},
		{1, "MOTOR_PLANE"},
		{2, "MOTOR_CONE"},
	};
	static const hkInternalClassEnum hkpRagdollConstraintDataEnums[] = {
		{"MotorIndex", hkpRagdollConstraintDataMotorIndexEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkpRagdollConstraintDataMotorIndexEnum = reinterpret_cast<const hkClassEnum*>(&hkpRagdollConstraintDataEnums[0]);
	static const hkInternalClassEnumItem hkpRagdollConstraintDataAtomsAxisEnumItems[] =
	{
		{0, "AXIS_TWIST"},
		{1, "AXIS_PLANES"},
		{2, "AXIS_CROSS_PRODUCT"},
	};
	static const hkInternalClassEnum hkpRagdollConstraintDataAtomsEnums[] = {
		{"Axis", hkpRagdollConstraintDataAtomsAxisEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkpRagdollConstraintDataAtomsAxisEnum = reinterpret_cast<const hkClassEnum*>(&hkpRagdollConstraintDataAtomsEnums[0]);
	static hkInternalClassMember hkpRagdollConstraintData_AtomsClass_Members[] =
	{
		{ "transforms", &hkpSetLocalTransformsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ragdollMotors", &hkpRagdollMotorConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angFriction", &hkpAngFrictionConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "twistLimit", &hkpTwistLimitConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "coneLimit", &hkpConeLimitConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "planesLimit", &hkpConeLimitConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ballSocket", &hkpBallSocketConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpRagdollConstraintDataAtomsClass(
		"hkpRagdollConstraintDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpRagdollConstraintDataAtomsEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpRagdollConstraintData_AtomsClass_Members),
		HK_COUNT_OF(hkpRagdollConstraintData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpRagdollConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpRagdollConstraintDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpRagdollConstraintDataClass(
		"hkpRagdollConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpRagdollConstraintDataEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpRagdollConstraintDataClass_Members),
		HK_COUNT_OF(hkpRagdollConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStiffSpringConstraintData_AtomsClass_Members[] =
	{
		{ "pivots", &hkpSetLocalTranslationsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "spring", &hkpStiffSpringConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpStiffSpringConstraintDataAtomsClass(
		"hkpStiffSpringConstraintDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStiffSpringConstraintData_AtomsClass_Members),
		HK_COUNT_OF(hkpStiffSpringConstraintData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStiffSpringConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpStiffSpringConstraintDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpStiffSpringConstraintDataClass(
		"hkpStiffSpringConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStiffSpringConstraintDataClass_Members),
		HK_COUNT_OF(hkpStiffSpringConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpWheelConstraintDataAtomsAxisEnumItems[] =
	{
		{0, "AXIS_SUSPENSION"},
		{1, "AXIS_PERP_SUSPENSION"},
		{0, "AXIS_AXLE"},
		{1, "AXIS_STEERING"},
	};
	static const hkInternalClassEnum hkpWheelConstraintDataAtomsEnums[] = {
		{"Axis", hkpWheelConstraintDataAtomsAxisEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkpWheelConstraintDataAtomsAxisEnum = reinterpret_cast<const hkClassEnum*>(&hkpWheelConstraintDataAtomsEnums[0]);
	static hkInternalClassMember hkpWheelConstraintData_AtomsClass_Members[] =
	{
		{ "suspensionBase", &hkpSetLocalTransformsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lin0Limit", &hkpLinLimitConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lin0Soft", &hkpLinSoftConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lin1", &hkpLinConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lin2", &hkpLinConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "steeringBase", &hkpSetLocalRotationsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "2dAng", &hkp2dAngConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpWheelConstraintDataAtomsClass(
		"hkpWheelConstraintDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpWheelConstraintDataAtomsEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpWheelConstraintData_AtomsClass_Members),
		HK_COUNT_OF(hkpWheelConstraintData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpWheelConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpWheelConstraintDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL },
		{ "initialAxleInB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initialSteeringAxisInB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpWheelConstraintDataClass(
		"hkpWheelConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpWheelConstraintDataClass_Members),
		HK_COUNT_OF(hkpWheelConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpRotationalConstraintData_AtomsClass_Members[] =
	{
		{ "rotations", &hkpSetLocalRotationsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "ang", &hkpAngConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpRotationalConstraintDataAtomsClass(
		"hkpRotationalConstraintDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpRotationalConstraintData_AtomsClass_Members),
		HK_COUNT_OF(hkpRotationalConstraintData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpRotationalConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpRotationalConstraintDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpRotationalConstraintDataClass(
		"hkpRotationalConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpRotationalConstraintDataClass_Members),
		HK_COUNT_OF(hkpRotationalConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBreakableConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpBridgeAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "constraintData", &hkpConstraintDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "childRuntimeSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "childNumSolverResults", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "solverResultLimit", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "removeWhenBroken", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "revertBackVelocityOnBreak", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "listener", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpBreakableConstraintDataClass(
		"hkpBreakableConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBreakableConstraintDataClass_Members),
		HK_COUNT_OF(hkpBreakableConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpConstraintChainDataClass(
		"hkpConstraintChainData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConstraintChainInstanceClass_Members[] =
	{
		{ "chainedEntities", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "action", &hkpConstraintChainInstanceActionClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConstraintChainInstanceClass(
		"hkpConstraintChainInstance",
		&hkpConstraintInstanceClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConstraintChainInstanceClass_Members),
		HK_COUNT_OF(hkpConstraintChainInstanceClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConstraintChainInstanceActionClass_Members[] =
	{
		{ "constraintInstance", &hkpConstraintChainInstanceClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConstraintChainInstanceActionClass(
		"hkpConstraintChainInstanceAction",
		&hkpActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConstraintChainInstanceActionClass_Members),
		HK_COUNT_OF(hkpConstraintChainInstanceActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBallSocketChainData_ConstraintInfoClass_Members[] =
	{
		{ "pivotInA", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pivotInB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpBallSocketChainDataConstraintInfoClass(
		"hkpBallSocketChainDataConstraintInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBallSocketChainData_ConstraintInfoClass_Members),
		HK_COUNT_OF(hkpBallSocketChainData_ConstraintInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBallSocketChainDataClass_Members[] =
	{
		{ "atoms", &hkpBridgeAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "infos", &hkpBallSocketChainDataConstraintInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "tau", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "cfm", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxErrorDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpBallSocketChainDataClass(
		"hkpBallSocketChainData",
		&hkpConstraintChainDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBallSocketChainDataClass_Members),
		HK_COUNT_OF(hkpBallSocketChainDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpHingeLimitsDataAtomsAxisEnumItems[] =
	{
		{0, "AXIS_AXLE"},
		{1, "AXIS_PERP_TO_AXLE_1"},
		{2, "AXIS_PERP_TO_AXLE_2"},
	};
	static const hkInternalClassEnum hkpHingeLimitsDataAtomsEnums[] = {
		{"Axis", hkpHingeLimitsDataAtomsAxisEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkpHingeLimitsDataAtomsAxisEnum = reinterpret_cast<const hkClassEnum*>(&hkpHingeLimitsDataAtomsEnums[0]);
	static hkInternalClassMember hkpHingeLimitsData_AtomsClass_Members[] =
	{
		{ "rotations", &hkpSetLocalRotationsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angLimit", &hkpAngLimitConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "2dAng", &hkp2dAngConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpHingeLimitsDataAtomsClass(
		"hkpHingeLimitsDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpHingeLimitsDataAtomsEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpHingeLimitsData_AtomsClass_Members),
		HK_COUNT_OF(hkpHingeLimitsData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpHingeLimitsDataClass_Members[] =
	{
		{ "atoms", &hkpHingeLimitsDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpHingeLimitsDataClass(
		"hkpHingeLimitsData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpHingeLimitsDataClass_Members),
		HK_COUNT_OF(hkpHingeLimitsDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPoweredChainData_ConstraintInfoClass_Members[] =
	{
		{ "pivotInA", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pivotInB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "aTc", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bTc", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "motors", &hkpConstraintMotorClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 3, 0, 0, HK_NULL },
		{ "switchBodies", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPoweredChainDataConstraintInfoClass(
		"hkpPoweredChainDataConstraintInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPoweredChainData_ConstraintInfoClass_Members),
		HK_COUNT_OF(hkpPoweredChainData_ConstraintInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPoweredChainDataClass_Members[] =
	{
		{ "atoms", &hkpBridgeAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "infos", &hkpPoweredChainDataConstraintInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "tau", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "cfmLinAdd", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "cfmLinMul", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "cfmAngAdd", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "cfmAngMul", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxErrorDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpPoweredChainData_DefaultStruct
		{
			int s_defaultOffsets[9];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_cfmLinAdd;
			hkReal m_cfmLinMul;
			hkReal m_cfmAngAdd;
			hkReal m_cfmAngMul;
		};
		const hkpPoweredChainData_DefaultStruct hkpPoweredChainData_Default =
		{
			{-1,-1,-1,-1,HK_OFFSET_OF(hkpPoweredChainData_DefaultStruct,m_cfmLinAdd),HK_OFFSET_OF(hkpPoweredChainData_DefaultStruct,m_cfmLinMul),HK_OFFSET_OF(hkpPoweredChainData_DefaultStruct,m_cfmAngAdd),HK_OFFSET_OF(hkpPoweredChainData_DefaultStruct,m_cfmAngMul),-1},
			0.1f*1.19209290e-07f,1.0f,0.1f*1.19209290e-07F,1.0f
		};
	}
	hkClass hkpPoweredChainDataClass(
		"hkpPoweredChainData",
		&hkpConstraintChainDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPoweredChainDataClass_Members),
		HK_COUNT_OF(hkpPoweredChainDataClass_Members),
		&hkpPoweredChainData_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpRagdollLimitsDataAtomsAxisEnumItems[] =
	{
		{0, "AXIS_TWIST"},
		{1, "AXIS_PLANES"},
		{2, "AXIS_CROSS_PRODUCT"},
	};
	static const hkInternalClassEnum hkpRagdollLimitsDataAtomsEnums[] = {
		{"Axis", hkpRagdollLimitsDataAtomsAxisEnumItems, 3, HK_NULL, 0 }
	};
	const hkClassEnum* hkpRagdollLimitsDataAtomsAxisEnum = reinterpret_cast<const hkClassEnum*>(&hkpRagdollLimitsDataAtomsEnums[0]);
	static hkInternalClassMember hkpRagdollLimitsData_AtomsClass_Members[] =
	{
		{ "rotations", &hkpSetLocalRotationsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "twistLimit", &hkpTwistLimitConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "coneLimit", &hkpConeLimitConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "planesLimit", &hkpConeLimitConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpRagdollLimitsDataAtomsClass(
		"hkpRagdollLimitsDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpRagdollLimitsDataAtomsEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpRagdollLimitsData_AtomsClass_Members),
		HK_COUNT_OF(hkpRagdollLimitsData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpRagdollLimitsDataClass_Members[] =
	{
		{ "atoms", &hkpRagdollLimitsDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpRagdollLimitsDataClass(
		"hkpRagdollLimitsData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpRagdollLimitsDataClass_Members),
		HK_COUNT_OF(hkpRagdollLimitsDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStiffSpringChainData_ConstraintInfoClass_Members[] =
	{
		{ "pivotInA", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pivotInB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "springLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpStiffSpringChainDataConstraintInfoClass(
		"hkpStiffSpringChainDataConstraintInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStiffSpringChainData_ConstraintInfoClass_Members),
		HK_COUNT_OF(hkpStiffSpringChainData_ConstraintInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpStiffSpringChainDataClass_Members[] =
	{
		{ "atoms", &hkpBridgeAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "infos", &hkpStiffSpringChainDataConstraintInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "tau", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "cfm", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpStiffSpringChainDataClass(
		"hkpStiffSpringChainData",
		&hkpConstraintChainDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpStiffSpringChainDataClass_Members),
		HK_COUNT_OF(hkpStiffSpringChainDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpGenericConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpBridgeAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "scheme", &hkpGenericConstraintDataSchemeClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpGenericConstraintDataClass(
		"hkpGenericConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpGenericConstraintDataClass_Members),
		HK_COUNT_OF(hkpGenericConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpGenericConstraintDataScheme_ConstraintInfoClass_Members[] =
	{
		{ "maxSizeOfSchema", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sizeOfSchemas", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numSolverResults", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numSolverElemTemps", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpGenericConstraintDataSchemeConstraintInfoClass(
		"hkpGenericConstraintDataSchemeConstraintInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpGenericConstraintDataScheme_ConstraintInfoClass_Members),
		HK_COUNT_OF(hkpGenericConstraintDataScheme_ConstraintInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpGenericConstraintDataSchemeClass_Members[] =
	{
		{ "info", &hkpGenericConstraintDataSchemeConstraintInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VECTOR4, 0, 0, 0, HK_NULL },
		{ "commands", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT32, 0, 0, 0, HK_NULL },
		{ "modifiers", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "motors", &hkpConstraintMotorClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpGenericConstraintDataSchemeClass(
		"hkpGenericConstraintDataScheme",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpGenericConstraintDataSchemeClass_Members),
		HK_COUNT_OF(hkpGenericConstraintDataSchemeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMalleableConstraintDataClass_Members[] =
	{
		{ "constraintData", &hkpConstraintDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "atoms", &hkpBridgeAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "strength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMalleableConstraintDataClass(
		"hkpMalleableConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMalleableConstraintDataClass_Members),
		HK_COUNT_OF(hkpMalleableConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpConstraintMotorMotorTypeEnumItems[] =
	{
		{0, "TYPE_INVALID"},
		{1, "TYPE_POSITION"},
		{2, "TYPE_VELOCITY"},
		{3, "TYPE_SPRING_DAMPER"},
		{4, "TYPE_CALLBACK"},
		{5, "TYPE_MAX"},
	};
	static const hkInternalClassEnum hkpConstraintMotorEnums[] = {
		{"MotorType", hkpConstraintMotorMotorTypeEnumItems, 6, HK_NULL, 0 }
	};
	const hkClassEnum* hkpConstraintMotorMotorTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpConstraintMotorEnums[0]);
	static hkInternalClassMember hkpConstraintMotorClass_Members[] =
	{
		{ "type", HK_NULL, hkpConstraintMotorMotorTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConstraintMotorClass(
		"hkpConstraintMotor",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpConstraintMotorEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpConstraintMotorClass_Members),
		HK_COUNT_OF(hkpConstraintMotorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpLimitedForceConstraintMotorClass_Members[] =
	{
		{ "minForce", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxForce", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpLimitedForceConstraintMotorClass(
		"hkpLimitedForceConstraintMotor",
		&hkpConstraintMotorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpLimitedForceConstraintMotorClass_Members),
		HK_COUNT_OF(hkpLimitedForceConstraintMotorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpCallbackConstraintMotorCallbackTypeEnumItems[] =
	{
		{0, "CALLBACK_MOTOR_TYPE_HAVOK_DEMO_SPRING_DAMPER"},
		{1, "CALLBACK_MOTOR_TYPE_USER_0"},
		{2, "CALLBACK_MOTOR_TYPE_USER_1"},
		{3, "CALLBACK_MOTOR_TYPE_USER_2"},
		{4, "CALLBACK_MOTOR_TYPE_USER_3"},
	};
	static const hkInternalClassEnum hkpCallbackConstraintMotorEnums[] = {
		{"CallbackType", hkpCallbackConstraintMotorCallbackTypeEnumItems, 5, HK_NULL, 0 }
	};
	const hkClassEnum* hkpCallbackConstraintMotorCallbackTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpCallbackConstraintMotorEnums[0]);
	static hkInternalClassMember hkpCallbackConstraintMotorClass_Members[] =
	{
		{ "callbackFunc", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "callbackType", HK_NULL, hkpCallbackConstraintMotorCallbackTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "userData0", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "userData1", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "userData2", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpCallbackConstraintMotorClass(
		"hkpCallbackConstraintMotor",
		&hkpLimitedForceConstraintMotorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpCallbackConstraintMotorEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpCallbackConstraintMotorClass_Members),
		HK_COUNT_OF(hkpCallbackConstraintMotorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPositionConstraintMotorClass_Members[] =
	{
		{ "tau", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "proportionalRecoveryVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "constantRecoveryVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPositionConstraintMotorClass(
		"hkpPositionConstraintMotor",
		&hkpLimitedForceConstraintMotorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPositionConstraintMotorClass_Members),
		HK_COUNT_OF(hkpPositionConstraintMotorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSpringDamperConstraintMotorClass_Members[] =
	{
		{ "springConstant", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "springDamping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSpringDamperConstraintMotorClass(
		"hkpSpringDamperConstraintMotor",
		&hkpLimitedForceConstraintMotorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSpringDamperConstraintMotorClass_Members),
		HK_COUNT_OF(hkpSpringDamperConstraintMotorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVelocityConstraintMotorClass_Members[] =
	{
		{ "tau", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "velocityTarget", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useVelocityTargetFromConstraintTargets", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVelocityConstraintMotorClass(
		"hkpVelocityConstraintMotor",
		&hkpLimitedForceConstraintMotorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVelocityConstraintMotorClass_Members),
		HK_COUNT_OF(hkpVelocityConstraintMotorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPulleyConstraintData_AtomsClass_Members[] =
	{
		{ "translations", &hkpSetLocalTranslationsConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pulley", &hkpPulleyConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPulleyConstraintDataAtomsClass(
		"hkpPulleyConstraintDataAtoms",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPulleyConstraintData_AtomsClass_Members),
		HK_COUNT_OF(hkpPulleyConstraintData_AtomsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPulleyConstraintDataClass_Members[] =
	{
		{ "atoms", &hkpPulleyConstraintDataAtomsClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::ALIGN_16, 0, HK_NULL }
	};
	hkClass hkpPulleyConstraintDataClass(
		"hkpPulleyConstraintData",
		&hkpConstraintDataClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPulleyConstraintDataClass_Members),
		HK_COUNT_OF(hkpPulleyConstraintDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpEntitySpuCollisionCallbackEventFilterEnumItems[] =
	{
		{0/*0x00*/, "SPU_SEND_NONE"},
		{1/*0x01*/, "SPU_SEND_CONTACT_POINT_ADDED"},
		{2/*0x02*/, "SPU_SEND_CONTACT_POINT_PROCESS"},
		{4/*0x04*/, "SPU_SEND_CONTACT_POINT_REMOVED"},
		{3/*SPU_SEND_CONTACT_POINT_ADDED|SPU_SEND_CONTACT_POINT_PROCESS*/, "SPU_SEND_CONTACT_POINT_ADDED_OR_PROCESS"},
	};
	static const hkInternalClassEnum hkpEntityEnums[] = {
		{"SpuCollisionCallbackEventFilter", hkpEntitySpuCollisionCallbackEventFilterEnumItems, 5, HK_NULL, 0 }
	};
	const hkClassEnum* hkpEntitySpuCollisionCallbackEventFilterEnum = reinterpret_cast<const hkClassEnum*>(&hkpEntityEnums[0]);
	static hkInternalClassMember hkpEntity_SmallArraySerializeOverrideTypeClass_Members[] =
	{
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "size", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "capacityAndFlags", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpEntitySmallArraySerializeOverrideTypeClass(
		"hkpEntitySmallArraySerializeOverrideType",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpEntity_SmallArraySerializeOverrideTypeClass_Members),
		HK_COUNT_OF(hkpEntity_SmallArraySerializeOverrideTypeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpEntity_SpuCollisionCallbackClass_Members[] =
	{
		{ "util", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "capacity", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "eventFilter", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "userFilter", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpEntitySpuCollisionCallbackClass(
		"hkpEntitySpuCollisionCallback",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpEntity_SpuCollisionCallbackClass_Members),
		HK_COUNT_OF(hkpEntity_SpuCollisionCallbackClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpEntity_ExtendedListenersClass_Members[] =
	{
		{ "activationListeners", &hkpEntitySmallArraySerializeOverrideTypeClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "entityListeners", &hkpEntitySmallArraySerializeOverrideTypeClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpEntityExtendedListenersClass(
		"hkpEntityExtendedListeners",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpEntity_ExtendedListenersClass_Members),
		HK_COUNT_OF(hkpEntity_ExtendedListenersClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpEntityClass_Members[] =
	{
		{ "material", &hkpMaterialClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "limitContactImpulseUtil", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "damageMultiplier", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "breakableBody", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "solverData", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "storageIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "processContactCallbackDelay", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "constraintsMaster", &hkpEntitySmallArraySerializeOverrideTypeClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "constraintsSlave", &hkpConstraintInstanceClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "constraintRuntime", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "simulationIsland", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "autoRemoveLevel", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numUserDatasInContactPointProperties", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "uid", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "spuCollisionCallback", &hkpEntitySpuCollisionCallbackClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extendedListeners", &hkpEntityExtendedListenersClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "motion", &hkpMaxSizeMotionClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionListeners", &hkpEntitySmallArraySerializeOverrideTypeClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "actions", &hkpEntitySmallArraySerializeOverrideTypeClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "localFrame", &hkLocalFrameClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpEntity_DefaultStruct
		{
			int s_defaultOffsets[20];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_damageMultiplier;
			hkUint32 m_uid;
		};
		const hkpEntity_DefaultStruct hkpEntity_Default =
		{
			{-1,-1,HK_OFFSET_OF(hkpEntity_DefaultStruct,m_damageMultiplier),-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkpEntity_DefaultStruct,m_uid),-1,-1,-1,-1,-1,-1},
			1,0xffffffff
		};
	}
	hkClass hkpEntityClass(
		"hkpEntity",
		&hkpWorldObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpEntityEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpEntityClass_Members),
		HK_COUNT_OF(hkpEntityClass_Members),
		&hkpEntity_Default,
		HK_NULL,
		0,
		0
		);
	hkClass hkpEntityDeactivatorClass(
		"hkpEntityDeactivator",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpFakeRigidBodyDeactivatorClass(
		"hkpFakeRigidBodyDeactivator",
		&hkpRigidBodyDeactivatorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpRigidBodyClass(
		"hkpRigidBody",
		&hkpEntityClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpRigidBodyDeactivatorDeactivatorTypeEnumItems[] =
	{
		{0, "DEACTIVATOR_INVALID"},
		{1, "DEACTIVATOR_NEVER"},
		{2, "DEACTIVATOR_SPATIAL"},
		{3, "DEACTIVATOR_MAX_ID"},
	};
	static const hkInternalClassEnum hkpRigidBodyDeactivatorEnums[] = {
		{"DeactivatorType", hkpRigidBodyDeactivatorDeactivatorTypeEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkpRigidBodyDeactivatorDeactivatorTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpRigidBodyDeactivatorEnums[0]);
	hkClass hkpRigidBodyDeactivatorClass(
		"hkpRigidBodyDeactivator",
		&hkpEntityDeactivatorClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpRigidBodyDeactivatorEnums),
		1,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSpatialRigidBodyDeactivator_SampleClass_Members[] =
	{
		{ "refPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "refRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSpatialRigidBodyDeactivatorSampleClass(
		"hkpSpatialRigidBodyDeactivatorSample",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSpatialRigidBodyDeactivator_SampleClass_Members),
		HK_COUNT_OF(hkpSpatialRigidBodyDeactivator_SampleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSpatialRigidBodyDeactivatorClass_Members[] =
	{
		{ "highFrequencySample", &hkpSpatialRigidBodyDeactivatorSampleClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "lowFrequencySample", &hkpSpatialRigidBodyDeactivatorSampleClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "radiusSqrd", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minHighFrequencyTranslation", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minHighFrequencyRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minLowFrequencyTranslation", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minLowFrequencyRotation", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSpatialRigidBodyDeactivatorClass(
		"hkpSpatialRigidBodyDeactivator",
		&hkpRigidBodyDeactivatorClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSpatialRigidBodyDeactivatorClass_Members),
		HK_COUNT_OF(hkpSpatialRigidBodyDeactivatorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpMotionMotionTypeEnumItems[] =
	{
		{0, "MOTION_INVALID"},
		{1, "MOTION_DYNAMIC"},
		{2, "MOTION_SPHERE_INERTIA"},
		{3, "MOTION_STABILIZED_SPHERE_INERTIA"},
		{4, "MOTION_BOX_INERTIA"},
		{5, "MOTION_STABILIZED_BOX_INERTIA"},
		{6, "MOTION_KEYFRAMED"},
		{7, "MOTION_FIXED"},
		{8, "MOTION_THIN_BOX_INERTIA"},
		{9, "MOTION_CHARACTER"},
		{10, "MOTION_MAX_ID"},
	};
	static const hkInternalClassEnum hkpMotionEnums[] = {
		{"MotionType", hkpMotionMotionTypeEnumItems, 11, HK_NULL, 0 }
	};
	const hkClassEnum* hkpMotionMotionTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpMotionEnums[0]);
	static hkInternalClassMember hkpMotionClass_Members[] =
	{
		{ "type", HK_NULL, hkpMotionMotionTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "deactivationIntegrateCounter", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deactivationNumInactiveFrames", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "motionState", &hkMotionStateClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "inertiaAndMassInv", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "linearVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "angularVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deactivationRefPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "deactivationRefOrientation", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "savedMotion", &hkpMaxSizeMotionClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "savedQualityTypeIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT16, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMotionClass(
		"hkpMotion",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpMotionEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpMotionClass_Members),
		HK_COUNT_OF(hkpMotionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpBoxMotionClass(
		"hkpBoxMotion",
		&hkpMotionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpCharacterMotionClass(
		"hkpCharacterMotion",
		&hkpMotionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpFixedRigidMotionClass(
		"hkpFixedRigidMotion",
		&hkpKeyframedRigidMotionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpKeyframedRigidMotionClass(
		"hkpKeyframedRigidMotion",
		&hkpMotionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpMaxSizeMotionClass(
		"hkpMaxSizeMotion",
		&hkpKeyframedRigidMotionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpSphereMotionClass(
		"hkpSphereMotion",
		&hkpMotionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpStabilizedBoxMotionClass(
		"hkpStabilizedBoxMotion",
		&hkpBoxMotionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpStabilizedSphereMotionClass(
		"hkpStabilizedSphereMotion",
		&hkpSphereMotionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpThinBoxMotionClass(
		"hkpThinBoxMotion",
		&hkpBoxMotionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpAabbPhantomClass_Members[] =
	{
		{ "aabb", &hkAabbClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "overlappingCollidables", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpAabbPhantomClass(
		"hkpAabbPhantom",
		&hkpPhantomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpAabbPhantomClass_Members),
		HK_COUNT_OF(hkpAabbPhantomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpCachingShapePhantomClass_Members[] =
	{
		{ "collisionDetails", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpCachingShapePhantomClass(
		"hkpCachingShapePhantom",
		&hkpShapePhantomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpCachingShapePhantomClass_Members),
		HK_COUNT_OF(hkpCachingShapePhantomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPhantomClass_Members[] =
	{
		{ "overlapListeners", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "phantomListeners", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpPhantomClass(
		"hkpPhantom",
		&hkpWorldObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPhantomClass_Members),
		HK_COUNT_OF(hkpPhantomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpShapePhantomClass_Members[] =
	{
		{ "motionState", &hkMotionStateClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpShapePhantomClass(
		"hkpShapePhantom",
		&hkpPhantomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpShapePhantomClass_Members),
		HK_COUNT_OF(hkpShapePhantomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSimpleShapePhantom_CollisionDetailClass_Members[] =
	{
		{ "collidable", &hkpCollidableClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSimpleShapePhantomCollisionDetailClass(
		"hkpSimpleShapePhantomCollisionDetail",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSimpleShapePhantom_CollisionDetailClass_Members),
		HK_COUNT_OF(hkpSimpleShapePhantom_CollisionDetailClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSimpleShapePhantomClass_Members[] =
	{
		{ "collisionDetails", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpSimpleShapePhantomClass(
		"hkpSimpleShapePhantom",
		&hkpShapePhantomClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSimpleShapePhantomClass_Members),
		HK_COUNT_OF(hkpSimpleShapePhantomClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPhysicsSystemClass_Members[] =
	{
		{ "rigidBodies", &hkpRigidBodyClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "constraints", &hkpConstraintInstanceClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "actions", &hkpActionClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "phantoms", &hkpPhantomClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "userData", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "active", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpPhysicsSystem_DefaultStruct
		{
			int s_defaultOffsets[7];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkBool m_active;
		};
		const hkpPhysicsSystem_DefaultStruct hkpPhysicsSystem_Default =
		{
			{-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkpPhysicsSystem_DefaultStruct,m_active)},
			true
		};
	}
	hkClass hkpPhysicsSystemClass(
		"hkpPhysicsSystem",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPhysicsSystemClass_Members),
		HK_COUNT_OF(hkpPhysicsSystemClass_Members),
		&hkpPhysicsSystem_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpWorldCinfoSolverTypeEnumItems[] =
	{
		{0, "SOLVER_TYPE_INVALID"},
		{1, "SOLVER_TYPE_2ITERS_SOFT"},
		{2, "SOLVER_TYPE_2ITERS_MEDIUM"},
		{3, "SOLVER_TYPE_2ITERS_HARD"},
		{4, "SOLVER_TYPE_4ITERS_SOFT"},
		{5, "SOLVER_TYPE_4ITERS_MEDIUM"},
		{6, "SOLVER_TYPE_4ITERS_HARD"},
		{7, "SOLVER_TYPE_8ITERS_SOFT"},
		{8, "SOLVER_TYPE_8ITERS_MEDIUM"},
		{9, "SOLVER_TYPE_8ITERS_HARD"},
		{10, "SOLVER_TYPE_MAX_ID"},
	};
	static const hkInternalClassEnumItem hkpWorldCinfoSimulationTypeEnumItems[] =
	{
		{0, "SIMULATION_TYPE_INVALID"},
		{1, "SIMULATION_TYPE_DISCRETE"},
		{2, "SIMULATION_TYPE_CONTINUOUS"},
		{3, "SIMULATION_TYPE_MULTITHREADED"},
	};
	static const hkInternalClassEnumItem hkpWorldCinfoContactPointGenerationEnumItems[] =
	{
		{0, "CONTACT_POINT_ACCEPT_ALWAYS"},
		{1, "CONTACT_POINT_REJECT_DUBIOUS"},
		{2, "CONTACT_POINT_REJECT_MANY"},
	};
	static const hkInternalClassEnumItem hkpWorldCinfoBroadPhaseBorderBehaviourEnumItems[] =
	{
		{0, "BROADPHASE_BORDER_ASSERT"},
		{1, "BROADPHASE_BORDER_FIX_ENTITY"},
		{2, "BROADPHASE_BORDER_REMOVE_ENTITY"},
		{3, "BROADPHASE_BORDER_DO_NOTHING"},
	};
	static const hkInternalClassEnum hkpWorldCinfoEnums[] = {
		{"SolverType", hkpWorldCinfoSolverTypeEnumItems, 11, HK_NULL, 0 },
		{"SimulationType", hkpWorldCinfoSimulationTypeEnumItems, 4, HK_NULL, 0 },
		{"ContactPointGeneration", hkpWorldCinfoContactPointGenerationEnumItems, 3, HK_NULL, 0 },
		{"BroadPhaseBorderBehaviour", hkpWorldCinfoBroadPhaseBorderBehaviourEnumItems, 4, HK_NULL, 0 }
	};
	const hkClassEnum* hkpWorldCinfoSolverTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpWorldCinfoEnums[0]);
	const hkClassEnum* hkpWorldCinfoSimulationTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpWorldCinfoEnums[1]);
	const hkClassEnum* hkpWorldCinfoContactPointGenerationEnum = reinterpret_cast<const hkClassEnum*>(&hkpWorldCinfoEnums[2]);
	const hkClassEnum* hkpWorldCinfoBroadPhaseBorderBehaviourEnum = reinterpret_cast<const hkClassEnum*>(&hkpWorldCinfoEnums[3]);
	static hkInternalClassMember hkpWorldCinfoClass_Members[] =
	{
		{ "gravity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "broadPhaseQuerySize", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contactRestingVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "broadPhaseBorderBehaviour", HK_NULL, hkpWorldCinfoBroadPhaseBorderBehaviourEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "broadPhaseWorldAabb", &hkAabbClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionFilter", &hkpCollisionFilterClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "convexListFilter", &hkpConvexListFilterClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "expectedMaxLinearVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sizeOfToiEventQueue", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "expectedMinPsiDeltaTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "memoryWatchDog", &hkWorldMemoryAvailableWatchDogClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "broadPhaseNumMarkers", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contactPointGeneration", HK_NULL, hkpWorldCinfoContactPointGenerationEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "allowToSkipConfirmedCallbacks", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "solverTau", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "solverDamp", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "solverIterations", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "solverMicrosteps", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "forceCoherentConstraintOrderingInSolver", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "snapCollisionToConvexEdgeThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "snapCollisionToConcaveEdgeThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "enableToiWeldRejection", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "enableDeprecatedWelding", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "iterativeLinearCastEarlyOutDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "iterativeLinearCastMaxIterations", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deactivationNumInactiveFramesSelectFlag0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deactivationNumInactiveFramesSelectFlag1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deactivationIntegrateCounter", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "shouldActivateOnRigidBodyTransformChange", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deactivationReferenceDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "toiCollisionResponseRotateNormal", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxSectorsPerCollideTask", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "processToisMultithreaded", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxEntriesPerToiCollideTask", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "enableDeactivation", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "simulationType", HK_NULL, hkpWorldCinfoSimulationTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "enableSimulationIslands", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minDesiredIslandSize", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "processActionsInSingleThread", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "frameMarkerPsiSnap", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpWorldCinfo_DefaultStruct
		{
			int s_defaultOffsets[41];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			_hkVector4 m_gravity;
			hkInt32 m_broadPhaseQuerySize;
			hkReal m_collisionTolerance;
			hkReal m_expectedMaxLinearVelocity;
			int m_sizeOfToiEventQueue;
			hkReal m_expectedMinPsiDeltaTime;
			hkReal m_solverDamp;
			hkInt32 m_solverIterations;
			hkInt32 m_solverMicrosteps;
			hkReal m_snapCollisionToConvexEdgeThreshold;
			hkReal m_snapCollisionToConcaveEdgeThreshold;
			hkReal m_iterativeLinearCastEarlyOutDistance;
			hkInt32 m_iterativeLinearCastMaxIterations;
			_hkBool m_shouldActivateOnRigidBodyTransformChange;
			hkReal m_deactivationReferenceDistance;
			hkReal m_toiCollisionResponseRotateNormal;
			int m_maxSectorsPerCollideTask;
			_hkBool m_processToisMultithreaded;
			int m_maxEntriesPerToiCollideTask;
			_hkBool m_enableDeactivation;
			_hkBool m_enableSimulationIslands;
			hkUint32 m_minDesiredIslandSize;
			_hkBool m_processActionsInSingleThread;
			hkReal m_frameMarkerPsiSnap;
		};
		const hkpWorldCinfo_DefaultStruct hkpWorldCinfo_Default =
		{
			{HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_gravity),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_broadPhaseQuerySize),-1,-1,-1,HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_collisionTolerance),-1,-1,HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_expectedMaxLinearVelocity),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_sizeOfToiEventQueue),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_expectedMinPsiDeltaTime),-1,-1,-1,-1,-1,HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_solverDamp),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_solverIterations),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_solverMicrosteps),-1,HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_snapCollisionToConvexEdgeThreshold),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_snapCollisionToConcaveEdgeThreshold),-1,-1,HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_iterativeLinearCastEarlyOutDistance),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_iterativeLinearCastMaxIterations),-1,-1,-1,HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_shouldActivateOnRigidBodyTransformChange),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_deactivationReferenceDistance),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_toiCollisionResponseRotateNormal),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_maxSectorsPerCollideTask),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_processToisMultithreaded),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_maxEntriesPerToiCollideTask),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_enableDeactivation),-1,HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_enableSimulationIslands),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_minDesiredIslandSize),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_processActionsInSingleThread),HK_OFFSET_OF(hkpWorldCinfo_DefaultStruct,m_frameMarkerPsiSnap)},
			{0,-9.8f,0},1024,.1f,200,250,1.0f/30.0f,.6f,4,1,.524f,0.698f,.01f,20,true,0.02f,0.2f,4,true,1,true,true,64,true,.0001f
		};
	}
	hkClass hkpWorldCinfoClass(
		"hkpWorldCinfo",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpWorldCinfoEnums),
		4,
		reinterpret_cast<const hkClassMember*>(hkpWorldCinfoClass_Members),
		HK_COUNT_OF(hkpWorldCinfoClass_Members),
		&hkpWorldCinfo_Default,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpWorldObjectBroadPhaseTypeEnumItems[] =
	{
		{0, "BROAD_PHASE_INVALID"},
		{1, "BROAD_PHASE_ENTITY"},
		{2, "BROAD_PHASE_PHANTOM"},
		{3, "BROAD_PHASE_BORDER"},
		{4, "BROAD_PHASE_MAX_ID"},
	};
	static const hkInternalClassEnum hkpWorldObjectEnums[] = {
		{"BroadPhaseType", hkpWorldObjectBroadPhaseTypeEnumItems, 5, HK_NULL, 0 }
	};
	const hkClassEnum* hkpWorldObjectBroadPhaseTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpWorldObjectEnums[0]);
	static hkInternalClassMember hkpWorldObjectClass_Members[] =
	{
		{ "world", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "userData", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collidable", &hkpLinkedCollidableClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "multiThreadCheck", &hkMultiThreadCheckClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "name", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "properties", &hkpPropertyClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpWorldObjectClass(
		"hkpWorldObject",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpWorldObjectEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpWorldObjectClass_Members),
		HK_COUNT_OF(hkpWorldObjectClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkWorldMemoryAvailableWatchDogClass_Members[] =
	{
		{ "minMemoryAvailable", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkWorldMemoryAvailableWatchDogClass(
		"hkWorldMemoryAvailableWatchDog",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkWorldMemoryAvailableWatchDogClass_Members),
		HK_COUNT_OF(hkWorldMemoryAvailableWatchDogClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpAgent1nSectorClass_Members[] =
	{
		{ "bytesAllocated", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pad0", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pad1", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pad2", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 512-16, 0, 0, HK_NULL }
	};
	hkClass hkpAgent1nSectorClass(
		"hkpAgent1nSector",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpAgent1nSectorClass_Members),
		HK_COUNT_OF(hkpAgent1nSectorClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpLinkedCollidableClass_Members[] =
	{
		{ "collisionEntries", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpLinkedCollidableClass(
		"hkpLinkedCollidable",
		&hkpCollidableClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpLinkedCollidableClass_Members),
		HK_COUNT_OF(hkpLinkedCollidableClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpBroadPhaseHandleClass_Members[] =
	{
		{ "id", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpBroadPhaseHandleClass(
		"hkpBroadPhaseHandle",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpBroadPhaseHandleClass_Members),
		HK_COUNT_OF(hkpBroadPhaseHandleClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConvexPieceStreamDataClass_Members[] =
	{
		{ "convexPieceStream", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "convexPieceOffsets", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL },
		{ "convexPieceSingleTriangles", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT32, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConvexPieceStreamDataClass(
		"hkpConvexPieceStreamData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConvexPieceStreamDataClass_Members),
		HK_COUNT_OF(hkpConvexPieceStreamDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMoppCodeReindexedTerminalClass_Members[] =
	{
		{ "origShapeKey", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "reindexedShapeKey", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMoppCodeReindexedTerminalClass(
		"hkpMoppCodeReindexedTerminal",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMoppCodeReindexedTerminalClass_Members),
		HK_COUNT_OF(hkpMoppCodeReindexedTerminalClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpMoppCodeBuildTypeEnumItems[] =
	{
		{0, "BUILT_WITH_CHUNK_SUBDIVISION"},
		{1, "BUILT_WITHOUT_CHUNK_SUBDIVISION"},
	};
	static const hkInternalClassEnum hkpMoppCodeEnums[] = {
		{"BuildType", hkpMoppCodeBuildTypeEnumItems, 2, HK_NULL, 0 }
	};
	const hkClassEnum* hkpMoppCodeBuildTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpMoppCodeEnums[0]);
	static hkInternalClassMember hkpMoppCode_CodeInfoClass_Members[] =
	{
		{ "offset", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMoppCodeCodeInfoClass(
		"hkpMoppCodeCodeInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMoppCode_CodeInfoClass_Members),
		HK_COUNT_OF(hkpMoppCode_CodeInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMoppCodeClass_Members[] =
	{
		{ "info", &hkpMoppCodeCodeInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "data", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "buildType", HK_NULL, hkpMoppCodeBuildTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMoppCodeClass(
		"hkpMoppCode",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpMoppCodeEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpMoppCodeClass_Members),
		HK_COUNT_OF(hkpMoppCodeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpAngularDashpotActionClass_Members[] =
	{
		{ "rotation", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "strength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpAngularDashpotActionClass(
		"hkpAngularDashpotAction",
		&hkpBinaryActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpAngularDashpotActionClass_Members),
		HK_COUNT_OF(hkpAngularDashpotActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpDashpotActionClass_Members[] =
	{
		{ "point", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL },
		{ "strength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "impulse", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpDashpotActionClass(
		"hkpDashpotAction",
		&hkpBinaryActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpDashpotActionClass_Members),
		HK_COUNT_OF(hkpDashpotActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMotorActionClass_Members[] =
	{
		{ "axis", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "spinRate", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "gain", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "active", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMotorActionClass(
		"hkpMotorAction",
		&hkpUnaryActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMotorActionClass_Members),
		HK_COUNT_OF(hkpMotorActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMouseSpringActionClass_Members[] =
	{
		{ "positionInRbLocal", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "mousePositionInWorld", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "springDamping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "springElasticity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxRelativeForce", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "objectDamping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "shapeKey", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "applyCallbacks", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpMouseSpringActionClass(
		"hkpMouseSpringAction",
		&hkpUnaryActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMouseSpringActionClass_Members),
		HK_COUNT_OF(hkpMouseSpringActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpReorientActionClass_Members[] =
	{
		{ "rotationAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "upAxis", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "strength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpReorientActionClass(
		"hkpReorientAction",
		&hkpUnaryActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpReorientActionClass_Members),
		HK_COUNT_OF(hkpReorientActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSpringActionClass_Members[] =
	{
		{ "lastForce", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "positionAinA", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "positionBinB", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "restLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "strength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "damping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "onCompression", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "onExtension", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSpringActionClass(
		"hkpSpringAction",
		&hkpBinaryActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSpringActionClass_Members),
		HK_COUNT_OF(hkpSpringActionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpCharacterProxyCinfoClass_Members[] =
	{
		{ "position", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "velocity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dynamicFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "staticFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "keepContactTolerance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "up", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extraUpStaticFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extraDownStaticFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "shapePhantom", &hkpShapePhantomClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "keepDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contactAngleSensitivity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "userPlanes", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxCharacterSpeedForSolver", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "characterStrength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "characterMass", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxSlope", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "penetrationRecoverySpeed", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxCastIterations", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "refreshManifoldInCheckSupport", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpCharacterProxyCinfo_DefaultStruct
		{
			int s_defaultOffsets[19];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_contactAngleSensitivity;
			int m_maxCastIterations;
		};
		const hkpCharacterProxyCinfo_DefaultStruct hkpCharacterProxyCinfo_Default =
		{
			{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkpCharacterProxyCinfo_DefaultStruct,m_contactAngleSensitivity),-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkpCharacterProxyCinfo_DefaultStruct,m_maxCastIterations),-1},
			10,10
		};
	}
	hkClass hkpCharacterProxyCinfoClass(
		"hkpCharacterProxyCinfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpCharacterProxyCinfoClass_Members),
		HK_COUNT_OF(hkpCharacterProxyCinfoClass_Members),
		&hkpCharacterProxyCinfo_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpConstrainedSystemFilterClass_Members[] =
	{
		{ "otherFilter", &hkpCollisionFilterClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpConstrainedSystemFilterClass(
		"hkpConstrainedSystemFilter",
		&hkpCollisionFilterClass,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpConstrainedSystemFilterClass_Members),
		HK_COUNT_OF(hkpConstrainedSystemFilterClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPairwiseCollisionFilter_CollisionPairClass_Members[] =
	{
		{ "a", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "b", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPairwiseCollisionFilterCollisionPairClass(
		"hkpPairwiseCollisionFilterCollisionPair",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPairwiseCollisionFilter_CollisionPairClass_Members),
		HK_COUNT_OF(hkpPairwiseCollisionFilter_CollisionPairClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPairwiseCollisionFilterClass_Members[] =
	{
		{ "disabledPairs", &hkpPairwiseCollisionFilterCollisionPairClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPairwiseCollisionFilterClass(
		"hkpPairwiseCollisionFilter",
		&hkpCollisionFilterClass,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPairwiseCollisionFilterClass_Members),
		HK_COUNT_OF(hkpPairwiseCollisionFilterClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPoweredChainMapper_TargetClass_Members[] =
	{
		{ "chain", &hkpPoweredChainDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "infoIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPoweredChainMapperTargetClass(
		"hkpPoweredChainMapperTarget",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPoweredChainMapper_TargetClass_Members),
		HK_COUNT_OF(hkpPoweredChainMapper_TargetClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPoweredChainMapper_LinkInfoClass_Members[] =
	{
		{ "firstTargetIdx", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numTargets", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "limitConstraint", &hkpConstraintInstanceClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPoweredChainMapperLinkInfoClass(
		"hkpPoweredChainMapperLinkInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPoweredChainMapper_LinkInfoClass_Members),
		HK_COUNT_OF(hkpPoweredChainMapper_LinkInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPoweredChainMapperClass_Members[] =
	{
		{ "links", &hkpPoweredChainMapperLinkInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "targets", &hkpPoweredChainMapperTargetClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "chains", &hkpConstraintChainInstanceClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPoweredChainMapperClass(
		"hkpPoweredChainMapper",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPoweredChainMapperClass_Members),
		HK_COUNT_OF(hkpPoweredChainMapperClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpDisableEntityCollisionFilterClass_Members[] =
	{
		{ "disabledEntities", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpDisableEntityCollisionFilterClass(
		"hkpDisableEntityCollisionFilter",
		&hkpCollisionFilterClass,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpDisableEntityCollisionFilterClass_Members),
		HK_COUNT_OF(hkpDisableEntityCollisionFilterClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpGroupCollisionFilterClass_Members[] =
	{
		{ "noGroupCollisionEnabled", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionGroups", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 32, 0, 0, HK_NULL }
	};
	hkClass hkpGroupCollisionFilterClass(
		"hkpGroupCollisionFilter",
		&hkpCollisionFilterClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpGroupCollisionFilterClass_Members),
		HK_COUNT_OF(hkpGroupCollisionFilterClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpMassPropertiesClass_Members[] =
	{
		{ "volume", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "mass", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "centerOfMass", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "inertiaTensor", HK_NULL, HK_NULL, hkClassMember::TYPE_MATRIX3, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpMassPropertiesClass(
		"hkpMassProperties",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpMassPropertiesClass_Members),
		HK_COUNT_OF(hkpMassPropertiesClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPhysicsSystemWithContactsClass_Members[] =
	{
		{ "contacts", &hkpSerializedAgentNnEntryClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPhysicsSystemWithContactsClass(
		"hkpPhysicsSystemWithContacts",
		&hkpPhysicsSystemClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPhysicsSystemWithContactsClass_Members),
		HK_COUNT_OF(hkpPhysicsSystemWithContactsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSerializedTrack1nInfoClass_Members[] =
	{
		{ "sectors", &hkpAgent1nSectorClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "subTracks", &hkpSerializedSubTrack1nInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSerializedTrack1nInfoClass(
		"hkpSerializedTrack1nInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSerializedTrack1nInfoClass_Members),
		HK_COUNT_OF(hkpSerializedTrack1nInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSerializedSubTrack1nInfoClass_Members[] =
	{
		{ "sectorIndex", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "offsetInSector", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSerializedSubTrack1nInfoClass(
		"hkpSerializedSubTrack1nInfo",
		&hkpSerializedTrack1nInfoClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSerializedSubTrack1nInfoClass_Members),
		HK_COUNT_OF(hkpSerializedSubTrack1nInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static const hkInternalClassEnumItem hkpSerializedAgentNnEntrySerializedAgentTypeEnumItems[] =
	{
		{0, "INVALID_AGENT_TYPE"},
		{1, "BOX_BOX_AGENT3"},
		{2, "CAPSULE_TRIANGLE_AGENT3"},
		{3, "PRED_GSK_AGENT3"},
		{4, "PRED_GSK_CYLINDER_AGENT3"},
		{5, "CONVEX_LIST_AGENT3"},
		{6, "LIST_AGENT3"},
		{7, "BV_TREE_AGENT3"},
		{8, "COLLECTION_COLLECTION_AGENT3"},
		{9, "COLLECTION_AGENT3"},
	};
	static const hkInternalClassEnum hkpSerializedAgentNnEntryEnums[] = {
		{"SerializedAgentType", hkpSerializedAgentNnEntrySerializedAgentTypeEnumItems, 10, HK_NULL, 0 }
	};
	const hkClassEnum* hkpSerializedAgentNnEntrySerializedAgentTypeEnum = reinterpret_cast<const hkClassEnum*>(&hkpSerializedAgentNnEntryEnums[0]);
	static hkInternalClassMember hkpSerializedAgentNnEntryClass_Members[] =
	{
		{ "bodyA", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "bodyB", &hkpEntityClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "bodyAId", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "bodyBId", HK_NULL, HK_NULL, hkClassMember::TYPE_ULONG, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "useEntityIds", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "agentType", HK_NULL, hkpSerializedAgentNnEntrySerializedAgentTypeEnum, hkClassMember::TYPE_ENUM, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "atom", &hkpSimpleContactConstraintAtomClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "propertiesStream", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "contactPoints", &hkContactPointClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "cpIdMgr", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_UINT8, 0, 0, 0, HK_NULL },
		{ "nnEntryData", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 128, 0, 0, HK_NULL },
		{ "trackInfo", &hkpSerializedTrack1nInfoClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "endianCheckBuffer", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT8, hkClassMember::TYPE_VOID, 4, 0, 0, HK_NULL },
		{ "version", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSerializedAgentNnEntryClass(
		"hkpSerializedAgentNnEntry",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassEnum*>(hkpSerializedAgentNnEntryEnums),
		1,
		reinterpret_cast<const hkClassMember*>(hkpSerializedAgentNnEntryClass_Members),
		HK_COUNT_OF(hkpSerializedAgentNnEntryClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpRigidBodyDisplayBindingClass_Members[] =
	{
		{ "rigidBody", &hkpRigidBodyClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "displayObject", &hkxMeshClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "rigidBodyFromDisplayObjectTransform", HK_NULL, HK_NULL, hkClassMember::TYPE_MATRIX4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpRigidBodyDisplayBindingClass(
		"hkpRigidBodyDisplayBinding",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpRigidBodyDisplayBindingClass_Members),
		HK_COUNT_OF(hkpRigidBodyDisplayBindingClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPhysicsSystemDisplayBindingClass_Members[] =
	{
		{ "bindings", &hkpRigidBodyDisplayBindingClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "system", &hkpPhysicsSystemClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPhysicsSystemDisplayBindingClass(
		"hkpPhysicsSystemDisplayBinding",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPhysicsSystemDisplayBindingClass_Members),
		HK_COUNT_OF(hkpPhysicsSystemDisplayBindingClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpDisplayBindingDataClass_Members[] =
	{
		{ "rigidBodyBindings", &hkpRigidBodyDisplayBindingClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL },
		{ "physicsSystemBindings", &hkpPhysicsSystemDisplayBindingClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpDisplayBindingDataClass(
		"hkpDisplayBindingData",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpDisplayBindingDataClass_Members),
		HK_COUNT_OF(hkpDisplayBindingDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpPhysicsDataClass_Members[] =
	{
		{ "worldCinfo", &hkpWorldCinfoClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "systems", &hkpPhysicsSystemClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpPhysicsDataClass(
		"hkpPhysicsData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpPhysicsDataClass_Members),
		HK_COUNT_OF(hkpPhysicsDataClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSerializedDisplayMarkerClass_Members[] =
	{
		{ "transform", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSerializedDisplayMarkerClass(
		"hkpSerializedDisplayMarker",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSerializedDisplayMarkerClass_Members),
		HK_COUNT_OF(hkpSerializedDisplayMarkerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSerializedDisplayMarkerListClass_Members[] =
	{
		{ "markers", &hkpSerializedDisplayMarkerClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSerializedDisplayMarkerListClass(
		"hkpSerializedDisplayMarkerList",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSerializedDisplayMarkerListClass_Members),
		HK_COUNT_OF(hkpSerializedDisplayMarkerListClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSerializedDisplayRbTransforms_DisplayTransformPairClass_Members[] =
	{
		{ "rb", &hkpRigidBodyClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "localToDisplay", HK_NULL, HK_NULL, hkClassMember::TYPE_TRANSFORM, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSerializedDisplayRbTransformsDisplayTransformPairClass(
		"hkpSerializedDisplayRbTransformsDisplayTransformPair",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSerializedDisplayRbTransforms_DisplayTransformPairClass_Members),
		HK_COUNT_OF(hkpSerializedDisplayRbTransforms_DisplayTransformPairClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpSerializedDisplayRbTransformsClass_Members[] =
	{
		{ "transforms", &hkpSerializedDisplayRbTransformsDisplayTransformPairClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpSerializedDisplayRbTransformsClass(
		"hkpSerializedDisplayRbTransforms",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpSerializedDisplayRbTransformsClass_Members),
		HK_COUNT_OF(hkpSerializedDisplayRbTransformsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleData_WheelComponentParamsClass_Members[] =
	{
		{ "radius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "mass", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "width", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "friction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "viscosityFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "slipAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "forceFeedbackMultiplier", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxContactBodyAcceleration", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "axle", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDataWheelComponentParamsClass(
		"hkpVehicleDataWheelComponentParams",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleData_WheelComponentParamsClass_Members),
		HK_COUNT_OF(hkpVehicleData_WheelComponentParamsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDataClass_Members[] =
	{
		{ "gravity", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numWheels", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "chassisOrientation", HK_NULL, HK_NULL, hkClassMember::TYPE_ROTATION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "torqueRollFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "torquePitchFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "torqueYawFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extraTorqueFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxVelocityForPositionalFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "chassisUnitInertiaYaw", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "chassisUnitInertiaRoll", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "chassisUnitInertiaPitch", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "frictionEqualizer", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "normalClippingAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxFrictionSolverMassRatio", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wheelParams", &hkpVehicleDataWheelComponentParamsClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "numWheelsPerAxle", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_INT8, 0, 0, 0, HK_NULL },
		{ "frictionDescription", &hkpVehicleFrictionDescriptionClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "chassisFrictionInertiaInvDiag", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "alreadyInitialised", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	namespace
	{
		struct hkpVehicleData_DefaultStruct
		{
			int s_defaultOffsets[19];
			typedef hkInt8 _hkBool;
			typedef hkReal _hkVector4[4];
			typedef hkReal _hkQuaternion[4];
			typedef hkReal _hkMatrix3[12];
			typedef hkReal _hkRotation[12];
			typedef hkReal _hkQsTransform[12];
			typedef hkReal _hkMatrix4[16];
			typedef hkReal _hkTransform[16];
			hkReal m_maxFrictionSolverMassRatio;
		};
		const hkpVehicleData_DefaultStruct hkpVehicleData_Default =
		{
			{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,HK_OFFSET_OF(hkpVehicleData_DefaultStruct,m_maxFrictionSolverMassRatio),-1,-1,-1,-1,-1},
			30.0
		};
	}
	hkClass hkpVehicleDataClass(
		"hkpVehicleData",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDataClass_Members),
		HK_COUNT_OF(hkpVehicleDataClass_Members),
		&hkpVehicleData_Default,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleInstance_WheelInfoClass_Members[] =
	{
		{ "contactPoint", &hkContactPointClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contactFriction", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "contactBody", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL },
		{ "contactShapeKey", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "hardPointWs", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rayEndPointWs", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentSuspensionLength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "suspensionDirectionWs", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "spinAxisChassisSpace", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "spinAxisWs", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "steeringOrientationChassisSpace", HK_NULL, HK_NULL, hkClassMember::TYPE_QUATERNION, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "spinVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "spinAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "skidEnergyDensity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sideForce", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "forwardSlipVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sideSlipVelocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleInstanceWheelInfoClass(
		"hkpVehicleInstanceWheelInfo",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleInstance_WheelInfoClass_Members),
		HK_COUNT_OF(hkpVehicleInstance_WheelInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleInstanceClass_Members[] =
	{
		{ "data", &hkpVehicleDataClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "driverInput", &hkpVehicleDriverInputClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "steering", &hkpVehicleSteeringClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "engine", &hkpVehicleEngineClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "transmission", &hkpVehicleTransmissionClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "brake", &hkpVehicleBrakeClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "suspension", &hkpVehicleSuspensionClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "aerodynamics", &hkpVehicleAerodynamicsClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "wheelCollide", &hkpVehicleWheelCollideClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "tyreMarks", &hkpTyremarksInfoClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "velocityDamper", &hkpVehicleVelocityDamperClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "wheelsInfo", &hkpVehicleInstanceWheelInfoClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "frictionStatus", &hkpVehicleFrictionStatusClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deviceStatus", &hkpVehicleDriverInputStatusClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "isFixed", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_BOOL, 0, 0, 0, HK_NULL },
		{ "wheelsTimeSinceMaxPedalInput", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tryingToReverse", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "torque", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "rpm", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "mainSteeringAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wheelsSteeringAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "isReversing", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "currentGear", HK_NULL, HK_NULL, hkClassMember::TYPE_INT8, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "delayed", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "clutchDelayCountdown", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleInstanceClass(
		"hkpVehicleInstance",
		&hkpUnaryActionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleInstanceClass_Members),
		HK_COUNT_OF(hkpVehicleInstanceClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpVehicleAerodynamicsClass(
		"hkpVehicleAerodynamics",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDefaultAerodynamicsClass_Members[] =
	{
		{ "airDensity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "frontalArea", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dragCoefficient", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "liftCoefficient", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "extraGravityws", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDefaultAerodynamicsClass(
		"hkpVehicleDefaultAerodynamics",
		&hkpVehicleAerodynamicsClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDefaultAerodynamicsClass_Members),
		HK_COUNT_OF(hkpVehicleDefaultAerodynamicsClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpVehicleBrakeClass(
		"hkpVehicleBrake",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDefaultBrake_WheelBrakingPropertiesClass_Members[] =
	{
		{ "maxBreakingTorque", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "minPedalInputToBlock", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "isConnectedToHandbrake", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDefaultBrakeWheelBrakingPropertiesClass(
		"hkpVehicleDefaultBrakeWheelBrakingProperties",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDefaultBrake_WheelBrakingPropertiesClass_Members),
		HK_COUNT_OF(hkpVehicleDefaultBrake_WheelBrakingPropertiesClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDefaultBrakeClass_Members[] =
	{
		{ "wheelBrakingProperties", &hkpVehicleDefaultBrakeWheelBrakingPropertiesClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "wheelsMinTimeToBlock", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDefaultBrakeClass(
		"hkpVehicleDefaultBrake",
		&hkpVehicleBrakeClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDefaultBrakeClass_Members),
		HK_COUNT_OF(hkpVehicleDefaultBrakeClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpVehicleDriverInputStatusClass(
		"hkpVehicleDriverInputStatus",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpVehicleDriverInputClass(
		"hkpVehicleDriverInput",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDriverInputAnalogStatusClass_Members[] =
	{
		{ "positionX", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "positionY", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "handbrakeButtonPressed", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "reverseButtonPressed", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDriverInputAnalogStatusClass(
		"hkpVehicleDriverInputAnalogStatus",
		&hkpVehicleDriverInputStatusClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDriverInputAnalogStatusClass_Members),
		HK_COUNT_OF(hkpVehicleDriverInputAnalogStatusClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDefaultAnalogDriverInputClass_Members[] =
	{
		{ "slopeChangePointX", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "initialSlope", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "deadZone", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "autoReverse", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDefaultAnalogDriverInputClass(
		"hkpVehicleDefaultAnalogDriverInput",
		&hkpVehicleDriverInputClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDefaultAnalogDriverInputClass_Members),
		HK_COUNT_OF(hkpVehicleDefaultAnalogDriverInputClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpVehicleEngineClass(
		"hkpVehicleEngine",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDefaultEngineClass_Members[] =
	{
		{ "minRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "optRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxTorque", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "torqueFactorAtMinRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "torqueFactorAtMaxRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "resistanceFactorAtMinRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "resistanceFactorAtOptRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "resistanceFactorAtMaxRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "clutchSlipRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDefaultEngineClass(
		"hkpVehicleDefaultEngine",
		&hkpVehicleEngineClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDefaultEngineClass_Members),
		HK_COUNT_OF(hkpVehicleDefaultEngineClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleFrictionDescription_AxisDescriptionClass_Members[] =
	{
		{ "frictionCircleYtab", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 16, 0, 0, HK_NULL },
		{ "xStep", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "xStart", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wheelSurfaceInertia", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wheelSurfaceInertiaInv", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wheelChassisMassRatio", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wheelRadius", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wheelRadiusInv", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wheelDownForceFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "wheelDownForceSumFactor", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleFrictionDescriptionAxisDescriptionClass(
		"hkpVehicleFrictionDescriptionAxisDescription",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleFrictionDescription_AxisDescriptionClass_Members),
		HK_COUNT_OF(hkpVehicleFrictionDescription_AxisDescriptionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleFrictionDescriptionClass_Members[] =
	{
		{ "wheelDistance", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "chassisMassInv", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "axleDescr", &hkpVehicleFrictionDescriptionAxisDescriptionClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleFrictionDescriptionClass(
		"hkpVehicleFrictionDescription",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleFrictionDescriptionClass_Members),
		HK_COUNT_OF(hkpVehicleFrictionDescriptionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleFrictionStatus_AxisStatusClass_Members[] =
	{
		{ "forward_slip_velocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "side_slip_velocity", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "skid_energy_density", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "side_force", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "delayed_forward_impulse", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "sideRhs", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "forwardRhs", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "relativeSideForce", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "relativeForwardForce", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleFrictionStatusAxisStatusClass(
		"hkpVehicleFrictionStatusAxisStatus",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleFrictionStatus_AxisStatusClass_Members),
		HK_COUNT_OF(hkpVehicleFrictionStatus_AxisStatusClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleFrictionStatusClass_Members[] =
	{
		{ "axis", &hkpVehicleFrictionStatusAxisStatusClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 2, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleFrictionStatusClass(
		"hkpVehicleFrictionStatus",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleFrictionStatusClass_Members),
		HK_COUNT_OF(hkpVehicleFrictionStatusClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpVehicleSteeringClass(
		"hkpVehicleSteering",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDefaultSteeringClass_Members[] =
	{
		{ "maxSteeringAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxSpeedFullSteeringAngle", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "doesWheelSteer", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_BOOL, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDefaultSteeringClass(
		"hkpVehicleDefaultSteering",
		&hkpVehicleSteeringClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDefaultSteeringClass_Members),
		HK_COUNT_OF(hkpVehicleDefaultSteeringClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleSuspension_SuspensionWheelParametersClass_Members[] =
	{
		{ "hardpointChassisSpace", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "directionChassisSpace", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "length", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleSuspensionSuspensionWheelParametersClass(
		"hkpVehicleSuspensionSuspensionWheelParameters",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleSuspension_SuspensionWheelParametersClass_Members),
		HK_COUNT_OF(hkpVehicleSuspension_SuspensionWheelParametersClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleSuspensionClass_Members[] =
	{
		{ "wheelParams", &hkpVehicleSuspensionSuspensionWheelParametersClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleSuspensionClass(
		"hkpVehicleSuspension",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleSuspensionClass_Members),
		HK_COUNT_OF(hkpVehicleSuspensionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDefaultSuspension_WheelSpringSuspensionParametersClass_Members[] =
	{
		{ "strength", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dampingCompression", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "dampingRelaxation", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDefaultSuspensionWheelSpringSuspensionParametersClass(
		"hkpVehicleDefaultSuspensionWheelSpringSuspensionParameters",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDefaultSuspension_WheelSpringSuspensionParametersClass_Members),
		HK_COUNT_OF(hkpVehicleDefaultSuspension_WheelSpringSuspensionParametersClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDefaultSuspensionClass_Members[] =
	{
		{ "wheelSpringParams", &hkpVehicleDefaultSuspensionWheelSpringSuspensionParametersClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDefaultSuspensionClass(
		"hkpVehicleDefaultSuspension",
		&hkpVehicleSuspensionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDefaultSuspensionClass_Members),
		HK_COUNT_OF(hkpVehicleDefaultSuspensionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpVehicleTransmissionClass(
		"hkpVehicleTransmission",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDefaultTransmissionClass_Members[] =
	{
		{ "downshiftRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "upshiftRPM", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "primaryTransmissionRatio", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "clutchDelayTime", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "reverseGearRatio", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "gearsRatio", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL },
		{ "wheelsTorqueRatio", HK_NULL, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_REAL, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDefaultTransmissionClass(
		"hkpVehicleDefaultTransmission",
		&hkpVehicleTransmissionClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDefaultTransmissionClass_Members),
		HK_COUNT_OF(hkpVehicleDefaultTransmissionClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpTyremarkPointClass_Members[] =
	{
		{ "pointLeft", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "pointRight", HK_NULL, HK_NULL, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpTyremarkPointClass(
		"hkpTyremarkPoint",
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpTyremarkPointClass_Members),
		HK_COUNT_OF(hkpTyremarkPointClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpTyremarksWheelClass_Members[] =
	{
		{ "currentPosition", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "numPoints", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tyremarkPoints", &hkpTyremarkPointClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL }
	};
	hkClass hkpTyremarksWheelClass(
		"hkpTyremarksWheel",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpTyremarksWheelClass_Members),
		HK_COUNT_OF(hkpTyremarksWheelClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpTyremarksInfoClass_Members[] =
	{
		{ "minTyremarkEnergy", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "maxTyremarkEnergy", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "tyremarksWheel", &hkpTyremarksWheelClass, HK_NULL, hkClassMember::TYPE_ARRAY, hkClassMember::TYPE_POINTER, 0, 0, 0, HK_NULL }
	};
	hkClass hkpTyremarksInfoClass(
		"hkpTyremarksInfo",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpTyremarksInfoClass_Members),
		HK_COUNT_OF(hkpTyremarksInfoClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	hkClass hkpVehicleVelocityDamperClass(
		"hkpVehicleVelocityDamper",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleDefaultVelocityDamperClass_Members[] =
	{
		{ "normalSpinDamping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionSpinDamping", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "collisionThreshold", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleDefaultVelocityDamperClass(
		"hkpVehicleDefaultVelocityDamper",
		&hkpVehicleVelocityDamperClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleDefaultVelocityDamperClass_Members),
		HK_COUNT_OF(hkpVehicleDefaultVelocityDamperClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleWheelCollideClass_Members[] =
	{
		{ "alreadyUsed", HK_NULL, HK_NULL, hkClassMember::TYPE_BOOL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleWheelCollideClass(
		"hkpVehicleWheelCollide",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleWheelCollideClass_Members),
		HK_COUNT_OF(hkpVehicleWheelCollideClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpRejectRayChassisListenerClass_Members[] =
	{
		{ "chassis", HK_NULL, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_VOID, 0, 0|hkClassMember::SERIALIZE_IGNORED, 0, HK_NULL }
	};
	hkClass hkpRejectRayChassisListenerClass(
		"hkpRejectRayChassisListener",
		&hkReferencedObjectClass,
		0,
		HK_NULL,
		1,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpRejectRayChassisListenerClass_Members),
		HK_COUNT_OF(hkpRejectRayChassisListenerClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);
	static hkInternalClassMember hkpVehicleRaycastWheelCollideClass_Members[] =
	{
		{ "wheelCollisionFilterInfo", HK_NULL, HK_NULL, hkClassMember::TYPE_UINT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL },
		{ "phantom", &hkpAabbPhantomClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "rejectRayChassisListener", &hkpRejectRayChassisListenerClass, HK_NULL, hkClassMember::TYPE_STRUCT, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	hkClass hkpVehicleRaycastWheelCollideClass(
		"hkpVehicleRaycastWheelCollide",
		&hkpVehicleWheelCollideClass,
		0,
		HK_NULL,
		0,
		HK_NULL,
		0,
		reinterpret_cast<const hkClassMember*>(hkpVehicleRaycastWheelCollideClass_Members),
		HK_COUNT_OF(hkpVehicleRaycastWheelCollideClass_Members),
		HK_NULL,
		HK_NULL,
		0,
		0
		);

	static hkClass* const Classes[] =
	{
		&hclActionClass,
		&hclBendLinkConstraintSetClass,
		&hclBendLinkConstraintSetLinkClass,
		&hclBlendSomeVerticesOperatorBlendEntryClass,
		&hclBlendSomeVerticesOperatorClass,
		&hclBonePlanesConstraintSetBonePlaneClass,
		&hclBonePlanesConstraintSetClass,
		&hclBufferDefinitionClass,
		&hclCapsuleShapeClass,
		&hclClothContainerClass,
		&hclClothDataClass,
		&hclClothStateBufferAccessClass,
		&hclClothStateClass,
		&hclCollidableClass,
		&hclConstraintSetClass,
		&hclConvexHeightFieldShapeClass,
		&hclCopyVerticesOperatorClass,
		&hclCylinderShapeClass,
		&hclGatherAllVerticesOperatorClass,
		&hclGatherSomeVerticesOperatorClass,
		&hclGatherSomeVerticesOperatorVertexPairClass,
		&hclHingeConstraintSetClass,
		&hclHingeConstraintSetHingeClass,
		&hclLocalRangeConstraintSetClass,
		&hclLocalRangeConstraintSetLocalConstraintClass,
		&hclMeshMeshDeformOperatorClass,
		&hclMeshMeshDeformOperatorTriangleVertexPairClass,
		&hclMoveParticlesOperatorClass,
		&hclMoveParticlesOperatorVertexParticlePairClass,
		&hclOperatorClass,
		&hclPlaneShapeClass,
		&hclRecalculateAllNormalsOperatorClass,
		&hclRecalculateSomeNormalsOperatorClass,
		&hclShapeClass,
		&hclSimClothDataClass,
		&hclSimClothDataCollisionPairClass,
		&hclSimClothDataParticleDataClass,
		&hclSimClothDataSimulationInfoClass,
		&hclSimClothPoseClass,
		&hclSimpleWindActionClass,
		&hclSimulateOperatorClass,
		&hclSkinOperatorBoneInfluenceClass,
		&hclSkinOperatorClass,
		&hclSphereShapeClass,
		&hclStandardLinkConstraintSetClass,
		&hclStandardLinkConstraintSetLinkClass,
		&hclStretchLinkConstraintSetClass,
		&hclStretchLinkConstraintSetLinkClass,
		&hclToolNamedObjectReferenceClass,
		&hclTransformSetDefinitionClass,
		&hclTransitionConstraintSetClass,
		&hclTransitionConstraintSetPerParticleClass,
		&hclTriangleSelectionInputClass,
		&hclVertexFloatInputClass,
		&hclVertexSelectionInputClass,
		&hclVolumeConstraintApplyDataClass,
		&hclVolumeConstraintClass,
		&hclVolumeConstraintFrameDataClass,
		&hkAabbClass,
		&hkAabbUint32Class,
		&hkBaseObjectClass,
		&hkBitFieldClass,
		&hkClassClass,
		&hkClassEnumClass,
		&hkClassEnumItemClass,
		&hkClassMemberClass,
		&hkColorClass,
		&hkContactPointClass,
		&hkContactPointMaterialClass,
		&hkCustomAttributesAttributeClass,
		&hkCustomAttributesClass,
		&hkLocalFrameClass,
		&hkMemoryResourceContainerClass,
		&hkMemoryResourceHandleClass,
		&hkMemoryResourceHandleExternalLinkClass,
		&hkMonitorStreamColorTableClass,
		&hkMonitorStreamColorTableColorPairClass,
		&hkMonitorStreamFrameInfoClass,
		&hkMonitorStreamStringMapClass,
		&hkMonitorStreamStringMapStringMapClass,
		&hkMoppBvTreeShapeBaseClass,
		&hkMotionStateClass,
		&hkMultiThreadCheckClass,
		&hkPackfileHeaderClass,
		&hkPackfileSectionHeaderClass,
		&hkReferencedObjectClass,
		&hkResourceBaseClass,
		&hkResourceContainerClass,
		&hkResourceHandleClass,
		&hkRootLevelContainerClass,
		&hkRootLevelContainerNamedVariantClass,
		&hkSimpleLocalFrameClass,
		&hkSphereClass,
		&hkSweptTransformClass,
		&hkVariableTweakingHelperBoolVariableInfoClass,
		&hkVariableTweakingHelperClass,
		&hkVariableTweakingHelperIntVariableInfoClass,
		&hkVariableTweakingHelperRealVariableInfoClass,
		&hkVariableTweakingHelperVector4VariableInfoClass,
		&hkWorldMemoryAvailableWatchDogClass,
		&hkaAnimatedReferenceFrameClass,
		&hkaAnimationBindingClass,
		&hkaAnimationClass,
		&hkaAnimationContainerClass,
		&hkaAnimationPreviewColorClass,
		&hkaAnimationPreviewColorContainerClass,
		&hkaAnnotationTrackAnnotationClass,
		&hkaAnnotationTrackClass,
		&hkaBoneAttachmentClass,
		&hkaBoneClass,
		&hkaDefaultAnimatedReferenceFrameClass,
		&hkaDeltaCompressedAnimationClass,
		&hkaDeltaCompressedAnimationQuantizationFormatClass,
		&hkaFootstepAnalysisInfoClass,
		&hkaFootstepAnalysisInfoContainerClass,
		&hkaInterleavedUncompressedAnimationClass,
		&hkaKeyFrameHierarchyUtilityClass,
		&hkaKeyFrameHierarchyUtilityControlDataClass,
		&hkaMeshBindingClass,
		&hkaMeshBindingMappingClass,
		&hkaRagdollInstanceClass,
		&hkaSkeletonClass,
		&hkaSkeletonLocalFrameOnBoneClass,
		&hkaSkeletonMapperClass,
		&hkaSkeletonMapperDataChainMappingClass,
		&hkaSkeletonMapperDataClass,
		&hkaSkeletonMapperDataSimpleMappingClass,
		&hkaSplineCompressedAnimationAnimationCompressionParamsClass,
		&hkaSplineCompressedAnimationClass,
		&hkaSplineCompressedAnimationTrackCompressionParamsClass,
		&hkaWaveletCompressedAnimationClass,
		&hkaWaveletCompressedAnimationCompressionParamsClass,
		&hkaWaveletCompressedAnimationQuantizationFormatClass,
		&hkbAlignBoneModifierClass,
		&hkbAnimatedSkeletonGeneratorClass,
		&hkbAttachmentModifierClass,
		&hkbAttachmentSetupClass,
		&hkbAttributeModifierAssignmentClass,
		&hkbAttributeModifierClass,
		&hkbBalanceControllerModifierClass,
		&hkbBalanceModifierClass,
		&hkbBalanceModifierStepInfoClass,
		&hkbBalanceRadialSelectorGeneratorClass,
		&hkbBehaviorGraphClass,
		&hkbBehaviorGraphDataClass,
		&hkbBehaviorGraphStringDataClass,
		&hkbBehaviorReferenceGeneratorClass,
		&hkbBlendCurveUtilsClass,
		&hkbBlenderGeneratorChildClass,
		&hkbBlenderGeneratorClass,
		&hkbBlendingTransitionEffectClass,
		&hkbBoolVariableSequencedDataClass,
		&hkbBoolVariableSequencedDataSampleClass,
		&hkbCatchFallModifierClass,
		&hkbCatchFallModifierHandClass,
		&hkbCharacterBoneInfoClass,
		&hkbCharacterClass,
		&hkbCharacterDataClass,
		&hkbCharacterSetupClass,
		&hkbCharacterStringDataClass,
		&hkbCheckBalanceModifierClass,
		&hkbCheckRagdollSpeedModifierClass,
		&hkbClimbMountingPredicateClass,
		&hkbClipGeneratorClass,
		&hkbClipTriggerClass,
		&hkbComputeDirectionModifierClass,
		&hkbComputeWorldFromModelModifierClass,
		&hkbConstrainRigidBodyModifierClass,
		&hkbContextClass,
		&hkbControlledReachModifierClass,
		&hkbCustomTestGeneratorClass,
		&hkbCustomTestGeneratorStruckClass,
		&hkbDampingModifierClass,
		&hkbDelayedModifierClass,
		&hkbDemoConfigCharacterInfoClass,
		&hkbDemoConfigClass,
		&hkbDemoConfigStickVariableInfoClass,
		&hkbDemoConfigTerrainInfoClass,
		&hkbDetectCloseToGroundModifierClass,
		&hkbEvaluateHandleModifierClass,
		&hkbEventClass,
		&hkbEventDrivenModifierClass,
		&hkbEventInfoClass,
		&hkbEventSequencedDataClass,
		&hkbEventSequencedDataSequencedEventClass,
		&hkbExtrapolatingTransitionEffectClass,
		&hkbFaceTargetModifierClass,
		&hkbFootIkControlDataClass,
		&hkbFootIkControlsModifierClass,
		&hkbFootIkGainsClass,
		&hkbFootIkModifierClass,
		&hkbFootIkModifierInternalLegDataClass,
		&hkbFootIkModifierLegClass,
		&hkbGeneratorClass,
		&hkbGeneratorOutputClass,
		&hkbGeneratorOutputConstTrackClass,
		&hkbGeneratorOutputListenerClass,
		&hkbGeneratorOutputTrackClass,
		&hkbGeneratorOutputTrackHeaderClass,
		&hkbGeneratorOutputTrackMasterHeaderClass,
		&hkbGeneratorOutputTracksClass,
		&hkbGeneratorTransitionEffectClass,
		&hkbGetHandleOnBoneModifierClass,
		&hkbGetUpModifierClass,
		&hkbGravityModifierClass,
		&hkbHandIkControlDataClass,
		&hkbHandIkControlsModifierClass,
		&hkbHandIkControlsModifierHandClass,
		&hkbHandIkModifierClass,
		&hkbHandIkModifierHandClass,
		&hkbHandleClass,
		&hkbHoldFromBlendingTransitionEffectClass,
		&hkbIntVariableSequencedDataClass,
		&hkbIntVariableSequencedDataSampleClass,
		&hkbJigglerGroupClass,
		&hkbJigglerModifierClass,
		&hkbKeyframeBonesModifierClass,
		&hkbLookAtModifierClass,
		&hkbManualSelectorGeneratorClass,
		&hkbMirrorModifierClass,
		&hkbMirroredSkeletonInfoClass,
		&hkbModifierClass,
		&hkbModifierGeneratorClass,
		&hkbModifierListClass,
		&hkbModifierWrapperClass,
		&hkbMoveBoneTowardTargetModifierClass,
		&hkbMoveCharacterModifierClass,
		&hkbNodeClass,
		&hkbPoseMatchingGeneratorClass,
		&hkbPoseStoringGeneratorOutputListenerClass,
		&hkbPoseStoringGeneratorOutputListenerStoredPoseClass,
		&hkbPositionRelativeSelectorGeneratorClass,
		&hkbPoweredRagdollControlDataClass,
		&hkbPoweredRagdollControlsModifierClass,
		&hkbPoweredRagdollModifierClass,
		&hkbPoweredRagdollModifierKeyframeInfoClass,
		&hkbPredicateClass,
		&hkbProjectDataClass,
		&hkbProjectStringDataClass,
		&hkbProxyModifierClass,
		&hkbProxyModifierProxyInfoClass,
		&hkbRadialSelectorGeneratorClass,
		&hkbRadialSelectorGeneratorGeneratorInfoClass,
		&hkbRadialSelectorGeneratorGeneratorPairClass,
		&hkbRagdollDriverModifierClass,
		&hkbRagdollForceModifierClass,
		&hkbReachModifierClass,
		&hkbReachModifierHandClass,
		&hkbReachTowardTargetModifierClass,
		&hkbReachTowardTargetModifierHandClass,
		&hkbRealVariableSequencedDataClass,
		&hkbRealVariableSequencedDataSampleClass,
		&hkbReferencePoseGeneratorClass,
		&hkbRegisteredGeneratorClass,
		&hkbRigidBodyRagdollControlDataClass,
		&hkbRigidBodyRagdollControlsModifierClass,
		&hkbRigidBodyRagdollModifierClass,
		&hkbRotateCharacterModifierClass,
		&hkbSenseHandleModifierClass,
		&hkbSequenceClass,
		&hkbSequenceStringDataClass,
		&hkbSequencedDataClass,
		&hkbSimpleCharacterClass,
		&hkbSplinePathGeneratorClass,
		&hkbStateDependentModifierClass,
		&hkbStateMachineActiveTransitionInfoClass,
		&hkbStateMachineClass,
		&hkbStateMachineNestedStateMachineDataClass,
		&hkbStateMachineProspectiveTransitionInfoClass,
		&hkbStateMachineStateInfoClass,
		&hkbStateMachineTimeIntervalClass,
		&hkbStateMachineTransitionInfoClass,
		&hkbStringPredicateClass,
		&hkbTargetClass,
		&hkbTargetRigidBodyModifierClass,
		&hkbTimerModifierClass,
		&hkbTransformVectorModifierClass,
		&hkbTransitionEffectClass,
		&hkbTwistModifierClass,
		&hkbVariableBindingSetBindingClass,
		&hkbVariableBindingSetClass,
		&hkbVariableInfoClass,
		&hkbVariableValueClass,
		&hkdBallGunBlueprintClass,
		&hkdBodyClass,
		&hkdBreakableBodyBlueprintClass,
		&hkdBreakableBodyClass,
		&hkdBreakableBodySmallArraySerializeOverrideTypeClass,
		&hkdBreakableShapeClass,
		&hkdBreakableShapeConnectionClass,
		&hkdChangeMassGunBlueprintClass,
		&hkdCompoundBreakableBodyBlueprintClass,
		&hkdCompoundBreakableShapeClass,
		&hkdContactRegionControllerClass,
		&hkdControllerClass,
		&hkdDeformableBreakableShapeClass,
		&hkdDeformationControllerClass,
		&hkdDestructionDemoConfigClass,
		&hkdFractureClass,
		&hkdGeometryClass,
		&hkdGeometryFaceClass,
		&hkdGeometryFaceIdentifierClass,
		&hkdGeometryObjectIdentifierClass,
		&hkdGeometryTriangleClass,
		&hkdGravityGunBlueprintClass,
		&hkdGrenadeGunBlueprintClass,
		&hkdMountedBallGunBlueprintClass,
		&hkdPieFractureClass,
		&hkdPropertiesClass,
		&hkdRandomSplitFractureClass,
		&hkdShapeClass,
		&hkdShapeInstanceInfoClass,
		&hkdShapeInstanceInfoRuntimeInfoClass,
		&hkdSliceFractureClass,
		&hkdSplitInHalfFractureClass,
		&hkdSplitShapeClass,
		&hkdWeaponBlueprintClass,
		&hkdWoodControllerClass,
		&hkdWoodFractureClass,
		&hkdWoodFractureSplittingDataClass,
		&hkp2dAngConstraintAtomClass,
		&hkpAabbPhantomClass,
		&hkpActionClass,
		&hkpAgent1nSectorClass,
		&hkpAngConstraintAtomClass,
		&hkpAngFrictionConstraintAtomClass,
		&hkpAngLimitConstraintAtomClass,
		&hkpAngMotorConstraintAtomClass,
		&hkpAngularDashpotActionClass,
		&hkpArrayActionClass,
		&hkpBallAndSocketConstraintDataAtomsClass,
		&hkpBallAndSocketConstraintDataClass,
		&hkpBallSocketChainDataClass,
		&hkpBallSocketChainDataConstraintInfoClass,
		&hkpBallSocketConstraintAtomClass,
		&hkpBinaryActionClass,
		&hkpBoxMotionClass,
		&hkpBoxShapeClass,
		&hkpBreakableConstraintDataClass,
		&hkpBridgeAtomsClass,
		&hkpBridgeConstraintAtomClass,
		&hkpBroadPhaseHandleClass,
		&hkpBvShapeClass,
		&hkpBvTreeShapeClass,
		&hkpCachingShapePhantomClass,
		&hkpCallbackConstraintMotorClass,
		&hkpCapsuleShapeClass,
		&hkpCdBodyClass,
		&hkpCharacterMotionClass,
		&hkpCharacterProxyCinfoClass,
		&hkpCollidableBoundingVolumeDataClass,
		&hkpCollidableClass,
		&hkpCollidableCollidableFilterClass,
		&hkpCollisionFilterClass,
		&hkpCollisionFilterListClass,
		&hkpCompressedSampledHeightFieldShapeClass,
		&hkpConeLimitConstraintAtomClass,
		&hkpConstrainedSystemFilterClass,
		&hkpConstraintAtomClass,
		&hkpConstraintChainDataClass,
		&hkpConstraintChainInstanceActionClass,
		&hkpConstraintChainInstanceClass,
		&hkpConstraintDataClass,
		&hkpConstraintInstanceClass,
		&hkpConstraintMotorClass,
		&hkpConvexListFilterClass,
		&hkpConvexListShapeClass,
		&hkpConvexPieceMeshShapeClass,
		&hkpConvexPieceStreamDataClass,
		&hkpConvexShapeClass,
		&hkpConvexTransformShapeBaseClass,
		&hkpConvexTransformShapeClass,
		&hkpConvexTranslateShapeClass,
		&hkpConvexVerticesConnectivityClass,
		&hkpConvexVerticesShapeClass,
		&hkpConvexVerticesShapeFourVectorsClass,
		&hkpCylinderShapeClass,
		&hkpDashpotActionClass,
		&hkpDefaultConvexListFilterClass,
		&hkpDisableEntityCollisionFilterClass,
		&hkpDisplayBindingDataClass,
		&hkpEntityClass,
		&hkpEntityDeactivatorClass,
		&hkpEntityExtendedListenersClass,
		&hkpEntitySmallArraySerializeOverrideTypeClass,
		&hkpEntitySpuCollisionCallbackClass,
		&hkpExtendedMeshShapeClass,
		&hkpExtendedMeshShapeShapesSubpartClass,
		&hkpExtendedMeshShapeSubpartClass,
		&hkpExtendedMeshShapeTrianglesSubpartClass,
		&hkpFakeRigidBodyDeactivatorClass,
		&hkpFastMeshShapeClass,
		&hkpFixedRigidMotionClass,
		&hkpGenericConstraintDataClass,
		&hkpGenericConstraintDataSchemeClass,
		&hkpGenericConstraintDataSchemeConstraintInfoClass,
		&hkpGroupCollisionFilterClass,
		&hkpGroupFilterClass,
		&hkpHeightFieldShapeClass,
		&hkpHingeConstraintDataAtomsClass,
		&hkpHingeConstraintDataClass,
		&hkpHingeLimitsDataAtomsClass,
		&hkpHingeLimitsDataClass,
		&hkpIgnoreModifierConstraintAtomClass,
		&hkpKeyframedRigidMotionClass,
		&hkpLimitedForceConstraintMotorClass,
		&hkpLimitedHingeConstraintDataAtomsClass,
		&hkpLimitedHingeConstraintDataClass,
		&hkpLinConstraintAtomClass,
		&hkpLinFrictionConstraintAtomClass,
		&hkpLinLimitConstraintAtomClass,
		&hkpLinMotorConstraintAtomClass,
		&hkpLinSoftConstraintAtomClass,
		&hkpLinearParametricCurveClass,
		&hkpLinkedCollidableClass,
		&hkpListShapeChildInfoClass,
		&hkpListShapeClass,
		&hkpMalleableConstraintDataClass,
		&hkpMassChangerModifierConstraintAtomClass,
		&hkpMassPropertiesClass,
		&hkpMaterialClass,
		&hkpMaxSizeMotionClass,
		&hkpMeshMaterialClass,
		&hkpMeshShapeClass,
		&hkpMeshShapeSubpartClass,
		&hkpModifierConstraintAtomClass,
		&hkpMoppBvTreeShapeClass,
		&hkpMoppCodeClass,
		&hkpMoppCodeCodeInfoClass,
		&hkpMoppCodeReindexedTerminalClass,
		&hkpMotionClass,
		&hkpMotorActionClass,
		&hkpMouseSpringActionClass,
		&hkpMovingSurfaceModifierConstraintAtomClass,
		&hkpMultiRayShapeClass,
		&hkpMultiRayShapeRayClass,
		&hkpMultiSphereShapeClass,
		&hkpNullCollisionFilterClass,
		&hkpOverwritePivotConstraintAtomClass,
		&hkpPairwiseCollisionFilterClass,
		&hkpPairwiseCollisionFilterCollisionPairClass,
		&hkpParametricCurveClass,
		&hkpPhantomCallbackShapeClass,
		&hkpPhantomClass,
		&hkpPhysicsDataClass,
		&hkpPhysicsSystemClass,
		&hkpPhysicsSystemDisplayBindingClass,
		&hkpPhysicsSystemWithContactsClass,
		&hkpPlaneShapeClass,
		&hkpPointToPathConstraintDataClass,
		&hkpPointToPlaneConstraintDataAtomsClass,
		&hkpPointToPlaneConstraintDataClass,
		&hkpPositionConstraintMotorClass,
		&hkpPoweredChainDataClass,
		&hkpPoweredChainDataConstraintInfoClass,
		&hkpPoweredChainMapperClass,
		&hkpPoweredChainMapperLinkInfoClass,
		&hkpPoweredChainMapperTargetClass,
		&hkpPrismaticConstraintDataAtomsClass,
		&hkpPrismaticConstraintDataClass,
		&hkpPropertyClass,
		&hkpPropertyValueClass,
		&hkpPulleyConstraintAtomClass,
		&hkpPulleyConstraintDataAtomsClass,
		&hkpPulleyConstraintDataClass,
		&hkpRagdollConstraintDataAtomsClass,
		&hkpRagdollConstraintDataClass,
		&hkpRagdollLimitsDataAtomsClass,
		&hkpRagdollLimitsDataClass,
		&hkpRagdollMotorConstraintAtomClass,
		&hkpRayCollidableFilterClass,
		&hkpRayShapeCollectionFilterClass,
		&hkpRejectRayChassisListenerClass,
		&hkpRemoveTerminalsMoppModifierClass,
		&hkpReorientActionClass,
		&hkpRigidBodyClass,
		&hkpRigidBodyDeactivatorClass,
		&hkpRigidBodyDisplayBindingClass,
		&hkpRotationalConstraintDataAtomsClass,
		&hkpRotationalConstraintDataClass,
		&hkpSampledHeightFieldShapeClass,
		&hkpSerializedAgentNnEntryClass,
		&hkpSerializedDisplayMarkerClass,
		&hkpSerializedDisplayMarkerListClass,
		&hkpSerializedDisplayRbTransformsClass,
		&hkpSerializedDisplayRbTransformsDisplayTransformPairClass,
		&hkpSerializedSubTrack1nInfoClass,
		&hkpSerializedTrack1nInfoClass,
		&hkpSetLocalRotationsConstraintAtomClass,
		&hkpSetLocalTransformsConstraintAtomClass,
		&hkpSetLocalTranslationsConstraintAtomClass,
		&hkpShapeClass,
		&hkpShapeCollectionClass,
		&hkpShapeCollectionFilterClass,
		&hkpShapeContainerClass,
		&hkpShapeInfoClass,
		&hkpShapePhantomClass,
		&hkpShapeRayCastInputClass,
		&hkpSimpleContactConstraintAtomClass,
		&hkpSimpleContactConstraintDataInfoClass,
		&hkpSimpleMeshShapeClass,
		&hkpSimpleMeshShapeTriangleClass,
		&hkpSimpleShapePhantomClass,
		&hkpSimpleShapePhantomCollisionDetailClass,
		&hkpSingleShapeContainerClass,
		&hkpSoftContactModifierConstraintAtomClass,
		&hkpSpatialRigidBodyDeactivatorClass,
		&hkpSpatialRigidBodyDeactivatorSampleClass,
		&hkpSphereMotionClass,
		&hkpSphereRepShapeClass,
		&hkpSphereShapeClass,
		&hkpSpringActionClass,
		&hkpSpringDamperConstraintMotorClass,
		&hkpStabilizedBoxMotionClass,
		&hkpStabilizedSphereMotionClass,
		&hkpStiffSpringChainDataClass,
		&hkpStiffSpringChainDataConstraintInfoClass,
		&hkpStiffSpringConstraintAtomClass,
		&hkpStiffSpringConstraintDataAtomsClass,
		&hkpStiffSpringConstraintDataClass,
		&hkpStorageExtendedMeshShapeClass,
		&hkpStorageExtendedMeshShapeMeshSubpartStorageClass,
		&hkpStorageExtendedMeshShapeShapeSubpartStorageClass,
		&hkpStorageMeshShapeClass,
		&hkpStorageMeshShapeSubpartStorageClass,
		&hkpStorageSampledHeightFieldShapeClass,
		&hkpThinBoxMotionClass,
		&hkpTransformShapeClass,
		&hkpTriSampledHeightFieldBvTreeShapeClass,
		&hkpTriSampledHeightFieldCollectionClass,
		&hkpTriangleShapeClass,
		&hkpTwistLimitConstraintAtomClass,
		&hkpTypedBroadPhaseHandleClass,
		&hkpTyremarkPointClass,
		&hkpTyremarksInfoClass,
		&hkpTyremarksWheelClass,
		&hkpUnaryActionClass,
		&hkpVehicleAerodynamicsClass,
		&hkpVehicleBrakeClass,
		&hkpVehicleDataClass,
		&hkpVehicleDataWheelComponentParamsClass,
		&hkpVehicleDefaultAerodynamicsClass,
		&hkpVehicleDefaultAnalogDriverInputClass,
		&hkpVehicleDefaultBrakeClass,
		&hkpVehicleDefaultBrakeWheelBrakingPropertiesClass,
		&hkpVehicleDefaultEngineClass,
		&hkpVehicleDefaultSteeringClass,
		&hkpVehicleDefaultSuspensionClass,
		&hkpVehicleDefaultSuspensionWheelSpringSuspensionParametersClass,
		&hkpVehicleDefaultTransmissionClass,
		&hkpVehicleDefaultVelocityDamperClass,
		&hkpVehicleDriverInputAnalogStatusClass,
		&hkpVehicleDriverInputClass,
		&hkpVehicleDriverInputStatusClass,
		&hkpVehicleEngineClass,
		&hkpVehicleFrictionDescriptionAxisDescriptionClass,
		&hkpVehicleFrictionDescriptionClass,
		&hkpVehicleFrictionStatusAxisStatusClass,
		&hkpVehicleFrictionStatusClass,
		&hkpVehicleInstanceClass,
		&hkpVehicleInstanceWheelInfoClass,
		&hkpVehicleRaycastWheelCollideClass,
		&hkpVehicleSteeringClass,
		&hkpVehicleSuspensionClass,
		&hkpVehicleSuspensionSuspensionWheelParametersClass,
		&hkpVehicleTransmissionClass,
		&hkpVehicleVelocityDamperClass,
		&hkpVehicleWheelCollideClass,
		&hkpVelocityConstraintMotorClass,
		&hkpViscousSurfaceModifierConstraintAtomClass,
		&hkpWeldingUtilityClass,
		&hkpWheelConstraintDataAtomsClass,
		&hkpWheelConstraintDataClass,
		&hkpWorldCinfoClass,
		&hkpWorldObjectClass,
		&hkxAnimatedFloatClass,
		&hkxAnimatedMatrixClass,
		&hkxAnimatedQuaternionClass,
		&hkxAnimatedVectorClass,
		&hkxAttributeClass,
		&hkxAttributeGroupClass,
		&hkxAttributeHolderClass,
		&hkxCameraClass,
		&hkxEdgeSelectionChannelClass,
		&hkxEnvironmentClass,
		&hkxEnvironmentVariableClass,
		&hkxIndexBufferClass,
		&hkxLightClass,
		&hkxMaterialClass,
		&hkxMaterialEffectClass,
		&hkxMaterialShaderClass,
		&hkxMaterialShaderSetClass,
		&hkxMaterialTextureStageClass,
		&hkxMeshClass,
		&hkxMeshSectionClass,
		&hkxMeshUserChannelInfoClass,
		&hkxNodeAnnotationDataClass,
		&hkxNodeClass,
		&hkxNodeSelectionSetClass,
		&hkxSceneClass,
		&hkxSkinBindingClass,
		&hkxSparselyAnimatedBoolClass,
		&hkxSparselyAnimatedEnumClass,
		&hkxSparselyAnimatedIntClass,
		&hkxSparselyAnimatedStringClass,
		&hkxSparselyAnimatedStringStringTypeClass,
		&hkxTextureFileClass,
		&hkxTextureInplaceClass,
		&hkxTriangleSelectionChannelClass,
		&hkxVertexBufferClass,
		&hkxVertexDescriptionClass,
		&hkxVertexDescriptionElementDeclClass,
		&hkxVertexFloatDataChannelClass,
		&hkxVertexIntDataChannelClass,
		&hkxVertexP4N4C1T10Class,
		&hkxVertexP4N4C1T2Class,
		&hkxVertexP4N4C1T6Class,
		&hkxVertexP4N4T4B4C1T10Class,
		&hkxVertexP4N4T4B4C1T2Class,
		&hkxVertexP4N4T4B4C1T6Class,
		&hkxVertexP4N4T4B4T4Class,
		&hkxVertexP4N4T4B4W4I4C1Q2Class,
		&hkxVertexP4N4T4B4W4I4C1T12Class,
		&hkxVertexP4N4T4B4W4I4C1T4Class,
		&hkxVertexP4N4T4B4W4I4C1T8Class,
		&hkxVertexP4N4T4B4W4I4Q4Class,
		&hkxVertexP4N4T4B4W4I4T6Class,
		&hkxVertexP4N4T4Class,
		&hkxVertexP4N4W4I4C1Q2Class,
		&hkxVertexP4N4W4I4C1T12Class,
		&hkxVertexP4N4W4I4C1T4Class,
		&hkxVertexP4N4W4I4C1T8Class,
		&hkxVertexP4N4W4I4T2Class,
		&hkxVertexP4N4W4I4T6Class,
		&hkxVertexSelectionChannelClass,
		&hkxVertexVectorDataChannelClass,
		HK_NULL
	}; 
	const hkStaticClassNameRegistry hkHavokDefaultClassRegistry
	(
		Classes,
		ClassVersion,
		VersionString
	);

} // namespace hkHavok610r1Classes

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
