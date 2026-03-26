#version 430 core

in vec3 velocity;
out vec4 FragColor;

vec3 velocityToColor(float t)
{
    const float s = 0.125;          
    if (t < s)    
        return mix(vec3(0,0,1), vec3(0,1,1), smoothstep(0.0, s, t));
    else if (t < 2*s)
        return mix(vec3(0,1,1), vec3(0,1,0), smoothstep(s, 2*s, t));
    else if (t < 3*s)
        return mix(vec3(0,1,0), vec3(1,1,0), smoothstep(2*s, 3*s, t));
    else if (t < 4*s)
        return mix(vec3(1,1,0), vec3(1,0,0), smoothstep(3*s, 4*s, t));
    else if (t < 5*s)
        return mix(vec3(1,0,0), vec3(1,0,1), smoothstep(4*s, 5*s, t));
    else if (t < 6*s)
        return mix(vec3(1,0,1), vec3(0,0,1), smoothstep(5*s, 6*s, t));
    else if (t < 7*s)
        return mix(vec3(0,0,1), vec3(0,1,0), smoothstep(6*s, 7*s, t));
    else               
        return mix(vec3(0,1,0), vec3(1,1,0), smoothstep(7*s, 1.0, t));
}

void main()
{
    vec2 centred = (gl_PointCoord - vec2(0.5)) * 2.0;
    float d2     = dot(centred, centred);
    float delta  = fwidth(sqrt(d2));
    float alpha  = 1.0 - smoothstep(1.0 - delta, 1.0 + delta, d2);

    // Carve out the circle
    if (alpha <= 0.0) discard;

    float speed = length(velocity);
    float t     = clamp(speed / 3.0, 0.0, 1.0);
    vec3  col   = velocityToColor(t);

    FragColor = vec4(col, alpha);
}