/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Deform/hkSkinOperator.h>
#include <Common/GeometryUtilities/Mesh/Deform/hkSkinOperatorGeneric.inl>

#if defined(HK_PLATFORM_RVL)
#include <Common/GeometryUtilities/Mesh/Deform/hkSkinOperatorWii.inl>
#endif

void hkSkinOperator::executeCpu(const Parameters& parameters)
{
	const hkUint32 skinningFlags	= parameters.m_input.m_skinExecutionFlags & parameters.m_output.m_skinExecutionFlags;
	const hkUint32 simdFlags		= parameters.m_input.m_simdExecutionFlags & parameters.m_output.m_simdExecutionFlags;

	if ( skinningFlags & HK_SKIN_POSITIONS )
	{
		if ( skinningFlags & HK_SKIN_NORMALS)
		{
			if ( (skinningFlags & HK_SKIN_ALL) == HK_SKIN_ALL )
			{
				// Do all
				if ( simdFlags == HK_USE_SIMD_FOR_ALL )
				{
					// Skin all with SIMD
					_skinGeneric <SKIN_SIMD_IN, SKIN_SIMD_OUT, 
						SKIN_SIMD_IN, SKIN_SIMD_OUT,
						SKIN_SIMD_IN, SKIN_SIMD_OUT,
						SKIN_SIMD_IN, SKIN_SIMD_OUT> (parameters);
					return;
				}

				if ( (simdFlags & HK_USE_SIMD_FOR_POSITIONS_AND_NORMALS) == HK_USE_SIMD_FOR_POSITIONS_AND_NORMALS )
				{
					// Special case for our d3d9s display buffers (input is all SIMD, output is SIMD+FLOAT)
					if ( (parameters.m_input.m_simdExecutionFlags & HK_USE_SIMD_FOR_POSITIONS_AND_NORMALS) == HK_USE_SIMD_FOR_POSITIONS_AND_NORMALS )
					{
						// Skin all. Use SIMD for position & normal, and only for input tangent and bitangent
						_skinGeneric <SKIN_SIMD_IN, SKIN_SIMD_OUT,
							SKIN_SIMD_IN, SKIN_SIMD_OUT,
							SKIN_SIMD_IN, SKIN_FLOAT32_OUT,
							SKIN_SIMD_IN, SKIN_FLOAT32_OUT> (parameters);
						return;
					}

					{
						// Skin all. Use SIMD only for position & normal
						_skinGeneric <SKIN_SIMD_IN, SKIN_SIMD_OUT,
							SKIN_SIMD_IN, SKIN_SIMD_OUT,
							SKIN_FLOAT32_IN, SKIN_FLOAT32_OUT,
							SKIN_FLOAT32_IN, SKIN_FLOAT32_OUT>  (parameters);
						return;
					}
				}

				{
					// Skin all. No SIMD
					_skinGeneric <SKIN_FLOAT32_IN, SKIN_FLOAT32_OUT,
						SKIN_FLOAT32_IN, SKIN_FLOAT32_OUT,
						SKIN_FLOAT32_IN, SKIN_FLOAT32_OUT,
						SKIN_FLOAT32_IN, SKIN_FLOAT32_OUT>  (parameters);
					return;
				}
			}

			// Do only positions and normals			
			{	
				//Special case for Wii.		
#if defined(HK_PLATFORM_RVL)
				{
					// If the locked cache is available for use and the buffer fits, use it, otherwise do generic skinning
					if ( !parameters.m_useLockedCache || !hkWiiSkinning_PN(parameters) )
					{
						_skinGeneric <SKIN_SIMD_IN, SKIN_SIMD_OUT,
							SKIN_SIMD_IN, SKIN_SIMD_OUT,
							SKIN_IGNORE, SKIN_IGNORE,
							SKIN_IGNORE, SKIN_IGNORE>  (parameters);
					}
					return;	
				}
#endif
				
				// Default case for other platforms.
				if ( simdFlags == HK_USE_SIMD_FOR_POSITIONS_AND_NORMALS )
				{
					_skinGeneric <SKIN_SIMD_IN, SKIN_SIMD_OUT,
						SKIN_SIMD_IN, SKIN_SIMD_OUT,
						SKIN_IGNORE, SKIN_IGNORE,
						SKIN_IGNORE, SKIN_IGNORE>  (parameters);
					return;
				}
				
				if ( simdFlags == HK_USE_SIMD_FOR_POSITIONS )
				{
					_skinGeneric <SKIN_SIMD_IN, SKIN_SIMD_OUT,
						SKIN_FLOAT32_IN, SKIN_FLOAT32_OUT,
						SKIN_IGNORE, SKIN_IGNORE,
						SKIN_IGNORE, SKIN_IGNORE>  (parameters);
					return;
				}
		
				{
					_skinGeneric <SKIN_FLOAT32_IN, SKIN_FLOAT32_OUT,
						SKIN_FLOAT32_IN, SKIN_FLOAT32_OUT,
						SKIN_IGNORE, SKIN_IGNORE,
						SKIN_IGNORE, SKIN_IGNORE>  (parameters);
					return;
				}
			}
		}

		// Do only positions
		{
			// Special case for Wii
#if defined(HK_PLATFORM_RVL)
			{
				// If the locked cache is available for use and the buffer fits, use it, otherwise do generic skinning
				if ( !parameters.m_useLockedCache || !hkWiiSkinning_P(parameters) )
				{
					_skinGeneric <SKIN_SIMD_IN, SKIN_SIMD_OUT,
						SKIN_IGNORE, SKIN_IGNORE,
						SKIN_IGNORE, SKIN_IGNORE,
						SKIN_IGNORE, SKIN_IGNORE>  (parameters);
				}
				return;	
			}
#endif
	
			// Default case for other platforms.
			if ( simdFlags & HK_USE_SIMD_FOR_POSITIONS )
			{
				_skinGeneric <SKIN_SIMD_IN, SKIN_SIMD_OUT,
					SKIN_IGNORE, SKIN_IGNORE,
					SKIN_IGNORE, SKIN_IGNORE,
					SKIN_IGNORE, SKIN_IGNORE>  (parameters);
				return;
			}

			{
				_skinGeneric <SKIN_FLOAT32_IN, SKIN_FLOAT32_OUT,
					SKIN_IGNORE, SKIN_IGNORE,
					SKIN_IGNORE, SKIN_IGNORE,
					SKIN_IGNORE, SKIN_IGNORE>  (parameters);
				return;
			}
		}
	}

	HK_WARN_ONCE (0x6ec4a445, "Invalid skin configuration - no positions?");
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
