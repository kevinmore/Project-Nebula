/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Base/Container/CommandStream/hkUnrollCaseMacro.h>
#include <Common/Visualize/Container/CommandStream/DebugCommands/hkDebugCommands.h>


void hkDebugCommandProcessor::exec( const hkCommand& command )
{
	const hkDebugCommand* debugCmd = (const hkDebugCommand*)&command;

	switch (debugCmd->m_secondaryType)
	{
	case hkDebugCommand::CMD_DEBUG_LINE:
		{
			const hkDebugLineCommand* c = (const hkDebugLineCommand*)debugCmd;
			HK_DISPLAY_LINE( c->m_start, c->m_end, c->m_color );
			break;
		}
	default:
		break;
	}
}

HK_FORCE_INLINE void hkDebugLineCommand::printCommand( hkOstream& out ) const
{
#ifndef HK_PLATFORM_SPU
	out << "hkDebugLineCommand Start=" << m_start << " End=" << m_end << " Color=" << m_color;
#endif
}


void hkDebugCommandProcessor::print( const hkCommand& command, hkOstream& stream ) const 
{
	switch (command.m_secondaryType)
	{
		HK_UNROLL_CASE_08(
		{						
			typedef hkDebugCommandTypeDiscriminator<UNROLL_I>::CommandType ct;		
			const ct* c = reinterpret_cast<const ct*>(&command);	
			c->printCommand(stream );
			break;
		}
		);
	}
	//  check if our unroll macro is sufficient by checking if command 33 falls back to our empty command
	{
		typedef hkDebugCommandTypeDiscriminator<9>::CommandType ct;	
		const ct* c = reinterpret_cast<const ct*>(&command);	
		c->checkIsEmptyCommand();		
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
