/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_PHYSICS_H
#define HKNP_PHYSICS_H

#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>

//
// Types used throughout the code
//

#include <Physics/Physics/hknpConfig.h>
#include <Physics/Physics/hknpTypes.h>

#include <Physics/Physics/Collide/Shape/hknpShape.h>
#include <Physics/Physics/Dynamics/Body/hknpBody.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotion.h>

#if !defined(HK_PLATFORM_SPU)
#	include <Physics/Physics/Dynamics/World/hknpWorld.h>
#else
#	include <Physics/Physics/Collide/hknpCdBody.h>
#	include <Physics/Physics/Dynamics/World/ModifierManager/hknpModifierManager.h>
#endif


#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpCompoundShape.h>

#endif // HKNP_PHYSICS_H

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
