#include "Simplex.h"

Simplex::Simplex()
	: m_curBits(0x0),
	  m_allBits(0x0)
{}

int Simplex::getPoints( QVector<vec3>& suppPointsA, QVector<vec3>& suppPointsB, QVector<vec3>& points ) const
{
	assert(suppPointsA.size() == 0 );
	assert(suppPointsB.size() == 0 );
	assert(points.size() == 0 );

	int count = 0;
	int i;
	uint bit;

	for ( i = 0, bit = 0x1; i < 4; ++i, bit <<= 1 ) 
	{
		if ( (m_curBits & bit) != 0 )
		{
			suppPointsA.push_back(m_suppPointsA[count]);
			suppPointsB.push_back(m_suppPointsB[count]);
			points.push_back(m_Points[count]);
			++count;
		}
	}

	return count;
}

void Simplex::addPoint( const vec3& point, const vec3& suppPointA, const vec3& suppPointB )
{
	assert(!isFull());

	m_lastFound = 0;
	m_lastBit = 0x1;

	// Look for the bit corresponding to one of the four point that is not in
	// the current simplex
	while ( (m_curBits & m_lastBit) != 0 )
	{
		++m_lastFound;
		m_lastBit <<= 1;
	}

	assert(m_lastFound >= 0 && m_lastFound < 4);

	// Add the point into the simplex
	m_Points[m_lastFound] = point;
	m_lengthSqrd[m_lastFound] = point.lengthSquared();
	m_allBits = m_curBits | m_lastBit;

	updateDiffLengths();    
	calcDeterminants();

	// Add the support points of objects A and B
	m_suppPointsA[m_lastFound] = suppPointA;
	m_suppPointsB[m_lastFound] = suppPointB;
}

bool Simplex::isDegenerate( const vec3& point ) const
{
	for ( int i = 0, bit = 0x1; i < 4; ++i, bit <<= 1 ) 
	{
		if ( ( (m_allBits & bit) != 0 ) && point == m_Points[i] ) 
			return true;
	}

	return false;
}

bool Simplex::isAffinelyIndependent() const
{
	float sum = 0.0;
	int i;
	uint bit;

	for ( i=0, bit = 0x1; i < 4; ++i, bit <<= 1 ) 
	{
		if ( (m_allBits & bit) != 0 )
		{
			sum += m_det[m_allBits][i];
		}
	}

	return (sum > 0.0);
}

void Simplex::closestPointAandB( vec3& pA, vec3& pB ) const
{
	float sum = 0.0;
	pA = vec3(0, 0, 0);
	pB = vec3(0, 0, 0);
	int i;
	uint bit;

	for ( i=0, bit=0x1; i<4; ++i, bit <<= 1 ) 
	{
		if ( (m_curBits & bit) != 0 )
		{
			sum += m_det[m_curBits][i];
			pA += m_det[m_curBits][i] * m_suppPointsA[i];
			pB += m_det[m_curBits][i] * m_suppPointsB[i];
		}
	}

	assert(sum > 0.0);
	float factor = 1.0f / sum;
	pA *= factor;
	pB *= factor;
}

// Run Johnson's Algorithm and compute 'v' which is a closest point to the origin in this simplex. 
// If this function succeeds, returns true. Otherwise, returns false.
bool Simplex::runJohnsonAlgorithm( vec3& v )
{
	uint subset;

	// Iterates all possible sub sets
	for ( subset = m_curBits; subset != 0x0; subset-- ) 
	{
		if ( isSubset(m_curBits, subset) && isValidSubset(subset | m_lastBit) ) 
		{
			m_curBits = subset | m_lastBit;   

			v = vec3(0, 0, 0);
			m_maxLengthSqrd = 0.0;
			float sum = 0.0;

			for ( int i=0, bit = 0x1; i < 4; ++i, bit <<= 1 ) 
			{
				if ( (m_curBits & bit) != 0 )
				{
					sum += m_det[m_curBits][i];
					v += m_det[m_curBits][i] * m_Points[i];

					if ( m_maxLengthSqrd < m_lengthSqrd[i] ) 
						m_maxLengthSqrd = m_lengthSqrd[i];
				}
			}

			assert(sum > 0.0);

			v = v / sum;
			return true;
		}
	}

	if ( isValidSubset(m_lastBit) ) 
	{
		m_curBits = m_lastBit;                  
		m_maxLengthSqrd = m_lengthSqrd[m_lastFound];
		v = m_Points[m_lastFound];                
		return true;
	}

	// Original GJK uses the backup procedure here.

	return false;
}

// To be a valid subset, all m_Det[subset][i] (i is part of 'subset') should be > 0 and 
// all other m_Det[subset][j] (j is not part of 'subset') should be <= 0. 
bool Simplex::isValidSubset( uint subset ) const
{
	int i;
	uint bit;

	for ( i = 0, bit = 0x1; i < 4; ++i, bit <<= 1 ) 
	{
		if ( (m_allBits & bit) != 0 )		
		{
			if ( (subset & bit) != 0 )
			{
				if ( m_det[subset][i] <= 0.0 )
					return false;
			}
			else if ( m_det[subset | bit][i] > 0.0 ) 
			{
				return false;
			}
		}
	}

	return true;
}

void Simplex::updateDiffLengths()
{
	int i;
	uint bit;

	for ( i=0, bit = 0x1; i < 4; ++i, bit <<= 1 ) 
	{
		if ( (m_curBits & bit) != 0 )
		{
			m_diffLength[i][m_lastFound] = m_Points[i] - m_Points[m_lastFound];
			m_diffLength[m_lastFound][i] = -m_diffLength[i][m_lastFound];
			m_diffLengthSqrd[i][m_lastFound] = m_diffLengthSqrd[m_lastFound][i] = m_diffLength[i][m_lastFound].lengthSquared();
		}
	}
}

void Simplex::calcDeterminants()
{
	m_det[m_lastBit][m_lastFound] = 1.0;

	if ( isEmpty() )
		return;

	for ( int i = 0, bitI = 0x1; i < 4; ++i, bitI <<= 1 ) 
	{
		if ( (m_curBits & bitI) != 0 )
		{
			uint bit2 = bitI | m_lastBit;

			m_det[bit2][i] = vec3::dotProduct(m_diffLength[m_lastFound][i], m_Points[m_lastFound]);
			m_det[bit2][m_lastFound] = vec3::dotProduct(m_diffLength[i][m_lastFound], m_Points[i]);

			for ( int j=0, bitJ = 0x1; j<i; ++j, bitJ <<= 1 ) 
			{
				if ( (m_curBits & bitJ) != 0 )
				{
					int k;
					uint bit3 = bitJ | bit2;

					k = m_diffLengthSqrd[i][j] < m_diffLengthSqrd[m_lastFound][j] ? i : m_lastFound;
					m_det[bit3][j] = m_det[bit2][i] * vec3::dotProduct(m_diffLength[k][j], m_Points[i]) +
						m_det[bit2][m_lastFound] * vec3::dotProduct(m_diffLength[k][j], m_Points[m_lastFound]);

					k = m_diffLengthSqrd[j][i] < m_diffLengthSqrd[m_lastFound][i] ? j : m_lastFound;
					m_det[bit3][i] = m_det[bitJ | m_lastBit][j] * vec3::dotProduct(m_diffLength[k][i], m_Points[j]) +
						m_det[bitJ | m_lastBit][m_lastFound] * vec3::dotProduct(m_diffLength[k][i], m_Points[m_lastFound]);

					k = m_diffLengthSqrd[i][m_lastFound] < m_diffLengthSqrd[j][m_lastFound] ? i : j;
					m_det[bit3][m_lastFound] = m_det[bitJ | bitI][j] * vec3::dotProduct(m_diffLength[k][m_lastFound], m_Points[j]) +
						m_det[bitJ | bitI][i] * vec3::dotProduct(m_diffLength[k][m_lastFound], m_Points[i]);
				}
			}
		}
	}

	if ( m_allBits == 0xf ) 
	{
		int k;

		k = m_diffLengthSqrd[1][0] < m_diffLengthSqrd[2][0] ? (m_diffLengthSqrd[1][0] < m_diffLengthSqrd[3][0] ? 1 : 3) : (m_diffLengthSqrd[2][0] < m_diffLengthSqrd[3][0] ? 2 : 3);
		m_det[0xf][0] = m_det[0xe][1] * vec3::dotProduct(m_diffLength[k][0], m_Points[1]) +
			m_det[0xe][2] * vec3::dotProduct(m_diffLength[k][0], m_Points[2]) +
			m_det[0xe][3] * vec3::dotProduct(m_diffLength[k][0], m_Points[3]);

		k = m_diffLengthSqrd[0][1] < m_diffLengthSqrd[2][1] ? (m_diffLengthSqrd[0][1] < m_diffLengthSqrd[3][1] ? 0 : 3) : (m_diffLengthSqrd[2][1] < m_diffLengthSqrd[3][1] ? 2 : 3);
		m_det[0xf][1] = m_det[0xd][0] * vec3::dotProduct(m_diffLength[k][1], m_Points[0]) +
			m_det[0xd][2] * vec3::dotProduct(m_diffLength[k][1], m_Points[2]) +
			m_det[0xd][3] * vec3::dotProduct(m_diffLength[k][1], m_Points[3]);

		k = m_diffLengthSqrd[0][2] < m_diffLengthSqrd[1][2] ? (m_diffLengthSqrd[0][2] < m_diffLengthSqrd[3][2] ? 0 : 3) : (m_diffLengthSqrd[1][2] < m_diffLengthSqrd[3][2] ? 1 : 3);
		m_det[0xf][2] = m_det[0xb][0] * vec3::dotProduct(m_diffLength[k][2], m_Points[0]) +
			m_det[0xb][1] * vec3::dotProduct(m_diffLength[k][2], m_Points[1]) +
			m_det[0xb][3] * vec3::dotProduct(m_diffLength[k][2], m_Points[3]);

		k = m_diffLengthSqrd[0][3] < m_diffLengthSqrd[1][3] ? (m_diffLengthSqrd[0][3] < m_diffLengthSqrd[2][3] ? 0 : 2) : (m_diffLengthSqrd[1][3] < m_diffLengthSqrd[2][3] ? 1 : 2);
		m_det[0xf][3] = m_det[0x7][0] * vec3::dotProduct(m_diffLength[k][3], m_Points[0]) +
			m_det[0x7][1] * vec3::dotProduct(m_diffLength[k][3], m_Points[1]) +
			m_det[0x7][2] * vec3::dotProduct(m_diffLength[k][3], m_Points[2]);
	}
}

