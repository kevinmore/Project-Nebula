/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Utilities/Dynamics/TimeSteppers/hkpAsynchronousTimestepper.h>


void HK_CALL hkAsynchronousTimestepper::stepAsynchronously( hkpWorld* world, hkReal frameDeltaTime, hkReal physicsDeltaTime )
{
	HK_WARN_ONCE(0xafe97523, "This utility is intended primarily for Havok demo use. If you wish to step the world asynchronously, you \
							  are encouraged to copy the code from this utility and integrate it into your game loop.");

	
	world->setFrameTimeMarker( frameDeltaTime );

	world->advanceTime();
	while ( !world->isSimulationAtMarker() ) 
	{
		HK_ASSERT( 0x11179564, world->isSimulationAtPsi() );

		{
			// Interact from game to physics
		}

		world->stepDeltaTime( physicsDeltaTime );
		
		if (world->isSimulationAtPsi() )
		{
			// Interact with physics: physics data to game data
		}
	}
	
}

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
