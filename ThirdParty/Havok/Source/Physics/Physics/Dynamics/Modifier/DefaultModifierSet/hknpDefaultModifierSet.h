/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_DEFAULT_MODIFIER_SET_H
#define HKNP_DEFAULT_MODIFIER_SET_H

#include <Physics/Physics/Dynamics/Modifier/EventCreators/ContactImpulseClipped/hknpContactImpulseClippedEventCreator.h>
#include <Physics/Physics/Dynamics/Modifier/EventCreators/ContactImpulse/hknpContactImpulseEventCreator.h>
#include <Physics/Physics/Dynamics/Modifier/MassChanger/hknpMassChangerModifier.h>
#include <Physics/Physics/Dynamics/Modifier/TriggerVolume/hknpTriggerVolumeModifier.h>
#include <Physics/Physics/Dynamics/Modifier/Restitution/hknpRestitutionModifier.h>
#include <Physics/Physics/Dynamics/Modifier/SoftContact/hknpSoftContactModifier.h>
#include <Physics/Physics/Dynamics/Modifier/SurfaceVelocity/hknpSurfaceVelocityModifier.h>
#include <Physics/Physics/Collide/Modifier/ManifoldEventCreator/hknpManifoldEventCreator.h>
#include <Physics/Physics/Collide/Modifier/Welding/hknpWeldingModifier.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/ConvexComposite/hknpConvexCompositeCollisionDetector.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/ConvexConvex/hknpConvexConvexCollisionDetector.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/CompositeComposite/hknpCompositeCompositeCollisionDetector.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/SignedDistanceField/hknpSignedDistanceFieldCollisionDetector.h>
#include <Physics/Physics/Dynamics/Modifier/EventCreators/ConstraintForce/hknpConstraintForceEventCreator.h>
#include <Physics/Physics/Dynamics/Modifier/EventCreators/ConstraintForceExceeded/hknpConstraintForceExceededEventCreator.h>


/// Helper class which stores a set of default modifiers used by the world constructor.
class hknpDefaultModifierSet : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		hknpManifoldEventCreator m_manifoldEventCreator;
		hknpContactImpulseEventCreator m_contactImpulseEventCreator;
		hknpContactImpulseClippedEventCreator m_clippedImpulseEventCreator;
		hknpConstraintForceEventCreator m_constraintForceEventCreator;
		hknpConstraintForceExceededEventCreator m_constraintForceExceededEventCreator;


		hknpNeighborWeldingModifier m_neighborWeldingModifier;
		hknpMotionWeldingModifier m_motionWeldingModifier;
		hknpTriangleWeldingModifier m_triangleWeldingModifier;

		hknpRestitutionModifier m_restitutionModifier;
		hknpSoftContactModifier m_softContactModifier;
		hknpTriggerVolumeModifier m_triggerVolumeModifier;
		hknpSurfaceVelocityModifier m_surfaceVelocityModifier;
		hknpMassChangerModifier m_massChangerModifier;

		hknpSetShapeKeyACdDetector m_setShakeKeyACdDetector;
		hknpConvexConvexCollisionDetector m_cvxCvxCdDetector;
		hknpConvexCompositeCollisionDetector m_cvxCompositeCdDetector;
		hknpCompositeCompositeCollisionDetector m_compositeCompositeCdDetector;
		hknpSignedDistanceFieldCollisionDetector m_signedDistanceFieldCdDetector;
};


#endif // HKNP_DEFAULT_MODIFIER_SET_H

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
