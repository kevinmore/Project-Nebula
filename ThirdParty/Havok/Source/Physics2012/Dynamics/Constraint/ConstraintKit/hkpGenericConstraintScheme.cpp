/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpGenericConstraintScheme.h>

#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpConstraintConstructionKit.h>

hkpGenericConstraintDataScheme::hkpGenericConstraintDataScheme(hkFinishLoadedObjectFlag f) : m_data(f), m_commands(f), m_modifiers(f), m_motors(f) 
{
	if (f.m_finishing)
	{
		m_info.clear();
		m_info.addHeader();
		hkpConstraintConstructionKit::computeConstraintInfo(m_commands, m_info);
		m_info.m_maxSizeOfSchema = m_info.m_sizeOfSchemas;
	}
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
