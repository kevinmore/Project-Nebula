/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/System/Error/hkError.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverVelocity.h>

HK_COMPILE_TIME_ASSERT( sizeof(hknpSolverVelocity) == 2*sizeof(hkVector4) );
HK_COMPILE_TIME_ASSERT( sizeof(hknpSolverSumVelocity) == 2*sizeof(hkVector4) );



/*static*/ HK_ALIGN16( const hkUint32 hknpSolverVelocity_Implementation::Masks::SetShiftMask[4] ) =		{ 0x00000000, 0x00000000, 0x00000000, 0x00000020 };
/*static*/ HK_ALIGN16( const hkUint32 hknpSolverVelocity_Implementation::Masks::SetPermMask[4] ) =		{ 0x00010203, 0x04050607, 0x08090A0B, 0x10111213 };
/*static*/ HK_ALIGN16( const hkUint32 hknpSolverVelocity_Implementation::Masks::GetAngPermMask[4] ) =	{ 0x0C0D0E0F, 0x10111213, 0x14151617, 0x18191A1B };

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
