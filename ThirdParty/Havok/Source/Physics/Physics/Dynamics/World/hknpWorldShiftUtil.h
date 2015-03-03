/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_WORLD_SHIFT_UTILITY_H
#define HKNP_WORLD_SHIFT_UTILITY_H

class hknpWorld;


/// Helper class for shifting a world.
class hknpWorldShiftUtil
{
	public:

		/// Shifts the whole world's coordinate by adding an offset to all positions.
		/// This should not be called during simulation.
		static void HK_CALL shiftWorld( hknpWorld* world, hkVector4Parameter offset );

		/// Shifts the broad phase to have it's center as close a possible to the requested position.
		/// This should not be called during simulation.
		static void HK_CALL shiftBroadPhase( hknpWorld* world, hkVector4Parameter requestedCenterPos, hkVector4& effectiveCenterPos, hkArray<hknpBodyId>* bodiesOutsideTheBroadPhase = HK_NULL);
};

#endif // HKNP_WORLD_SHIFT_UTILITY_H

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
