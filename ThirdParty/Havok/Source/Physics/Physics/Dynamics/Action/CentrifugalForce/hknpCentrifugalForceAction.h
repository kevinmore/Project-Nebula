/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CentrifugalForce_ACTION_H
#define HKNP_CentrifugalForce_ACTION_H

#include <Physics/Physics/Dynamics/Action/hknpAction.h>
//#include <Common/Base/Reflection/Attributes/hkAttributes.h>


/// A simple action, which implements a correct motion integration.
/// For stability reasons, the physics engine uses a simplified integration scheme which ignores
/// centrifugal forces. However in some cases you really care about those forces. In this case you attach
/// an instance of this action to the desired body.
class hknpCentrifugalForceAction: public hknpUnaryAction
{
	public:
		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS);

		HK_DECLARE_REFLECTION();

		/// Creates a CentrifugalForce.
		hknpCentrifugalForceAction( hknpBodyId idA );

		// Serializing constructor
		hknpCentrifugalForceAction( class hkFinishLoadedObjectFlag flag ) : hknpUnaryAction(flag) {}

		// hknpAction interface implementation
		virtual ApplyActionResult applyAction( const hknpSimulationThreadContext& tl, const hknpSolverInfo& stepInfo, hknpCdPairWriter* HK_RESTRICT pairWriter );

		virtual void onShiftWorld( hkVector4Parameter offset ) { }
};

#endif // HKNP_CentrifugalForce_ACTION_H

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
